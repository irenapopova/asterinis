from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asterinis.exceptions import RoutingError

from .registry import NLPProviderProfile, NLPProviderRegistry


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
    ) -> None:
        self.registry = registry or NLPProviderRegistry()

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
    ) -> NLPProviderSelection:
        task = task.strip().lower()
        if not task:
            raise ValueError("task cannot be empty.")
        if max_cost is not None and max_cost < 0:
            raise ValueError("max_cost cannot be negative.")
        if max_latency_ms is not None and max_latency_ms < 0:
            raise ValueError("max_latency_ms cannot be negative.")

        language = language.strip().lower() if language else None
        candidates: list[NLPProviderProfile] = []

        for profile in self.registry.profiles():
            if not profile.enabled or task not in profile.capabilities:
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

        candidates.sort(
            key=lambda item: (-item.quality, item.cost, item.latency_ms)
        )
        selected = candidates[0]

        return NLPProviderSelection(
            provider=selected.name,
            capability=task,
            quality=selected.quality,
            cost=selected.cost,
            latency_ms=selected.latency_ms,
            reason=(
                "Selected the highest-quality eligible provider, "
                "then preferred lower cost and latency."
            ),
            metadata={
                "candidate_count": len(candidates),
                "language": language,
            },
        )

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
        profile = self.registry.get(selection.provider)
        result = profile.provider.analyze(text, **kwargs)

        if hasattr(result, "metadata"):
            result.metadata.update(
                {
                    "provider": selection.provider,
                    "task": selection.capability,
                    "routing": selection.metadata,
                }
            )

        return result
