from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asterinis.exceptions import RoutingError

from .health import HealthRegistry


@dataclass(slots=True)
class ProviderProfile:
    name: str
    cost: float
    priority: int = 0
    capabilities: set[str] = field(
        default_factory=set
    )
    expected_latency_seconds: float | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()

        if not self.name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        if self.cost < 0:
            raise ValueError(
                "Provider cost cannot be negative."
            )

        if (
            self.expected_latency_seconds is not None
            and self.expected_latency_seconds < 0
        ):
            raise ValueError(
                "Expected latency cannot be negative."
            )


@dataclass(slots=True)
class ProviderSelection:
    provider: str
    cost: float
    reason: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class CostAwareRouter:
    """
    Selects an available provider that satisfies the requested capability,
    preferring lower cost and then higher priority.
    """

    def __init__(
        self,
        *,
        health: HealthRegistry | None = None,
    ) -> None:
        self.health = health or HealthRegistry()
        self._providers: dict[
            str,
            ProviderProfile,
        ] = {}

    def register(
        self,
        profile: ProviderProfile,
    ) -> None:
        self._providers[profile.name] = profile

    def select(
        self,
        capability: str,
    ) -> ProviderSelection:
        capability = capability.strip()

        if not capability:
            raise ValueError(
                "capability cannot be empty."
            )

        candidates = [
            profile
            for profile in self._providers.values()
            if capability in profile.capabilities
            and self.health.available(profile.name)
        ]

        if not candidates:
            raise RoutingError(
                f"No available provider supports '{capability}'."
            )

        candidates.sort(
            key=lambda profile: (
                profile.cost,
                -profile.priority,
            )
        )

        selected = candidates[0]

        return ProviderSelection(
            provider=selected.name,
            cost=selected.cost,
            reason=(
                "Selected the lowest-cost available "
                "provider with the required capability."
            ),
            metadata={
                "capability": capability,
                "candidate_count": len(candidates),
            },
        )