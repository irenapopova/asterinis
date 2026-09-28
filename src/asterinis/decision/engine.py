from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from asterinis.exceptions import RoutingError
from asterinis.learning import StrategyRecord, StrategyStore

from .policies import DecisionPolicy
from .result import Decision


Handler = Callable[[str], Any]


@dataclass(slots=True)
class ProviderOption:
    name: str
    capability: str
    handler: Handler
    quality: float = 0.5
    cost: float = 0.0
    latency_ms: float = 0.0
    external: bool = False


class DecisionEngine:
    """Explainable provider selection across NLP, RAG, agents, and LLMs."""

    def __init__(self, *, learning_store: StrategyStore | None = None) -> None:
        self.learning_store = learning_store or StrategyStore()
        self._options: dict[str, ProviderOption] = {}

    def register(self, option: ProviderOption, *, replace: bool = False) -> None:
        if not 0.0 <= option.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1.")
        if option.cost < 0 or option.latency_ms < 0:
            raise ValueError("cost and latency_ms cannot be negative.")
        if option.name in self._options and not replace:
            raise ValueError(f"Provider '{option.name}' is already registered.")
        self._options[option.name] = option

    def decide(
        self,
        capability: str,
        *,
        policy: DecisionPolicy | None = None,
    ) -> Decision:
        capability = capability.strip().lower()
        if not capability:
            raise ValueError("capability cannot be empty.")
        policy = policy or DecisionPolicy()
        candidates = [
            option
            for option in self._options.values()
            if option.capability == capability
            and option.cost <= (policy.max_cost if policy.max_cost is not None else float("inf"))
            and option.latency_ms <= (policy.max_latency_ms if policy.max_latency_ms is not None else float("inf"))
            and (policy.allow_external or not option.external)
            and not (policy.sensitive and option.external)
        ]
        if policy.mode == "local":
            candidates = [option for option in candidates if not option.external]
        if not candidates:
            raise RoutingError(f"No provider satisfies policy for '{capability}'.")

        def score(option: ProviderOption) -> float:
            history = tuple(
                record
                for record in self.learning_store.for_strategy(option.name)
                if record.query_type == capability
            )
            learned = option.quality
            if history:
                success = sum(record.success for record in history) / len(history)
                quality = [record.quality_score for record in history if record.quality_score is not None]
                learned = (success + (sum(quality) / len(quality) if quality else option.quality)) / 2
            if policy.mode == "best_quality":
                return learned
            if policy.mode == "local":
                return learned + (1.0 if not option.external else 0.0)
            return learned - min(1.0, option.cost) * 0.25 - min(1.0, option.latency_ms / 1000) * 0.1

        ranked = sorted(candidates, key=score, reverse=True)
        selected = ranked[0]
        selected_score = score(selected)
        return Decision(
            provider=selected.name,
            capability=capability,
            score=selected_score,
            reason=(
                f"Selected {selected.name} for {capability} using "
                f"{policy.mode} policy; quality, cost, latency, and history were considered."
            ),
            estimated_cost=selected.cost,
            estimated_latency_ms=selected.latency_ms,
            metadata={
                "mode": policy.mode,
                "candidate_count": len(candidates),
                "external": selected.external,
            },
        )

    def execute(
        self,
        text: str,
        capability: str,
        *,
        policy: DecisionPolicy | None = None,
    ) -> tuple[Decision, Any]:
        decision = self.decide(capability, policy=policy)
        option = self._options[decision.provider]
        started = perf_counter()
        try:
            output = option.handler(text)
        except Exception:
            self.learning_store.add(
                StrategyRecord(
                    strategy=option.name,
                    query_type=capability,
                    success=False,
                    latency_seconds=perf_counter() - started,
                    cost=option.cost,
                )
            )
            raise
        self.learning_store.add(
            StrategyRecord(
                strategy=option.name,
                query_type=capability,
                success=True,
                latency_seconds=perf_counter() - started,
                cost=option.cost,
            )
        )
        return decision, output
