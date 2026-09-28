from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from asterinis.exceptions import RoutingError

from .registry import NLPProviderProfile, NLPProviderRegistry
from asterinis.learning import StrategyRecord, StrategyScorer, StrategyStore


@dataclass(slots=True)
class NLPProviderSelection:
    provider: str
    capability: str
    quality: float
    cost: float
    latency_ms: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class NLPTaskRouter:
    """Select and invoke an NLP provider for a requested task."""

    def __init__(
        self,
        registry: NLPProviderRegistry | None = None,
        *,
        learning_store: StrategyStore | None = None,
        learning_scorer: StrategyScorer | None = None,
        minimum_learning_samples: int = 1,
    ) -> None:
        self.registry = registry or NLPProviderRegistry()
        if minimum_learning_samples < 1:
            raise ValueError(
                "minimum_learning_samples must be greater than zero."
            )
        self.learning_store = learning_store or StrategyStore()
        self.learning_scorer = learning_scorer or StrategyScorer()
        self.minimum_learning_samples = minimum_learning_samples

    def record_feedback(
        self,
        provider: str,
        task: str,
        *,
        success: bool,
        quality: float | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one provider outcome for future routing decisions."""
        provider = provider.strip()
        task = task.strip().lower()

        if not provider:
            raise ValueError("provider cannot be empty.")
        if not task:
            raise ValueError("task cannot be empty.")

        self.learning_store.add(
            StrategyRecord(
                strategy=provider,
                query_type=task,
                success=success,
                quality_score=quality,
                latency_seconds=(
                    latency_ms / 1000.0
                    if latency_ms is not None
                    else None
                ),
                cost=cost,
                confidence=confidence,
                metadata=metadata or {},
            )
        )

    def register(
        self,
        name: str,
        provider: Any,
        *,
        capabilities: set[str],
        languages: set[str] | None = None,
        quality: float = 0.5,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        replace: bool = False,
    ) -> None:
        if not callable(getattr(provider, "analyze", None)):
            raise TypeError("provider must expose an analyze() method.")

        self.registry.register(
            NLPProviderProfile(
                name=name,
                provider=provider,
                capabilities=capabilities,
                languages=languages or set(),
                quality=quality,
                cost=cost,
                latency_ms=latency_ms,
            ),
            replace=replace,
        )

    def select(
        self,
        task: str,
        *,
        language: str | None = None,
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
        exclude: set[str] | None = None,
    ) -> NLPProviderSelection:
        task = task.strip().lower()
        if not task:
            raise ValueError("task cannot be empty.")
        if max_cost is not None and max_cost < 0:
            raise ValueError("max_cost cannot be negative.")
        if max_latency_ms is not None and max_latency_ms < 0:
            raise ValueError("max_latency_ms cannot be negative.")

        language = language.strip().lower() if language else None
        excluded = exclude or set()
        candidates: list[NLPProviderProfile] = []

        for profile in self.registry.profiles():
            if (
                not profile.enabled
                or profile.name in excluded
                or task not in profile.capabilities
            ):
                continue
            if language and profile.languages and language not in profile.languages:
                continue
            if max_cost is not None and profile.cost > max_cost:
                continue
            if max_latency_ms is not None and profile.latency_ms > max_latency_ms:
                continue
            candidates.append(profile)

        if not candidates:
            raise RoutingError(
                f"No available NLP provider supports '{task}'."
            )

        scored_candidates: list[tuple[NLPProviderProfile, float, int]] = []
        for candidate in candidates:
            history = tuple(
                record
                for record in self.learning_store.for_strategy(candidate.name)
                if record.query_type == task
            )
            if len(history) >= self.minimum_learning_samples:
                learned_score = self.learning_scorer.score(
                    candidate.name,
                    history,
                ).score
                effective_quality = (
                    candidate.quality + learned_score
                ) / 2.0
                sample_count = len(history)
            else:
                effective_quality = candidate.quality
                sample_count = 0
            scored_candidates.append(
                (candidate, effective_quality, sample_count)
            )

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].cost,
                item[0].latency_ms,
            )
        )
        selected, effective_quality, sample_count = scored_candidates[0]

        return NLPProviderSelection(
            provider=selected.name,
            capability=task,
            quality=effective_quality,
            cost=selected.cost,
            latency_ms=selected.latency_ms,
            reason=(
                "Selected the highest-quality eligible provider, "
                "then preferred lower cost and latency."
            ),
            metadata={
                "candidate_count": len(candidates),
                "language": language,
                "learning_samples": sample_count,
            },
        )

    def invoke_selection(
        self,
        selection: NLPProviderSelection,
        text: str,
        **kwargs: Any,
    ) -> Any:
        profile = self.registry.get(selection.provider)
        started_at = perf_counter()

        try:
            result = profile.provider.analyze(text, **kwargs)
        except Exception:
            self.record_feedback(
                selection.provider,
                selection.capability,
                success=False,
                latency_ms=(perf_counter() - started_at) * 1000.0,
                cost=profile.cost,
            )
            raise

        confidence = getattr(result, "confidence", None)
        self.record_feedback(
            selection.provider,
            selection.capability,
            success=True,
            quality=confidence,
            confidence=confidence,
            latency_ms=(perf_counter() - started_at) * 1000.0,
            cost=profile.cost,
        )

        if hasattr(result, "metadata"):
            result.metadata.update(
                {
                    "provider": selection.provider,
                    "task": selection.capability,
                    "routing": selection.metadata,
                }
            )

        return result

    def analyze(
        self,
        text: str,
        *,
        task: str = "classification",
        language: str | None = None,
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text cannot be empty.")

        selection = self.select(
            task,
            language=language,
            max_cost=max_cost,
            max_latency_ms=max_latency_ms,
        )
        return self.invoke_selection(selection, text, **kwargs)
