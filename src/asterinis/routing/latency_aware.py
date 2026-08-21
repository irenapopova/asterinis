from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asterinis.exceptions import RoutingError

from .health import HealthRegistry


@dataclass(slots=True)
class LatencyProfile:
    """
    Runtime latency information used for provider selection.
    """

    provider: str
    expected_latency_seconds: float
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = self.provider.strip()

        if not self.provider:
            raise ValueError("provider cannot be empty.")

        if self.expected_latency_seconds < 0:
            raise ValueError(
                "expected_latency_seconds cannot be negative."
            )


@dataclass(slots=True)
class LatencySelection:
    provider: str
    expected_latency_seconds: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "expected_latency_seconds": (
                self.expected_latency_seconds
            ),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class LatencyAwareRouter:
    """
    Selects the fastest available provider.

    The router can use static latency estimates and, when available, the
    latest observed latency stored in HealthRegistry.
    """

    def __init__(
        self,
        *,
        health: HealthRegistry | None = None,
    ) -> None:
        self.health = health or HealthRegistry()
        self._profiles: dict[str, LatencyProfile] = {}

    def register(
        self,
        profile: LatencyProfile,
        *,
        replace: bool = False,
    ) -> None:
        if (
            profile.provider in self._profiles
            and not replace
        ):
            raise ValueError(
                f"Provider '{profile.provider}' is already registered."
            )

        self._profiles[profile.provider] = profile

    def select(
        self,
        *,
        maximum_latency_seconds: float | None = None,
    ) -> LatencySelection:
        if (
            maximum_latency_seconds is not None
            and maximum_latency_seconds < 0
        ):
            raise ValueError(
                "maximum_latency_seconds cannot be negative."
            )

        candidates: list[
            tuple[LatencyProfile, float]
        ] = []

        for profile in self._profiles.values():
            if not self.health.available(profile.provider):
                continue

            observed = self.health.get(profile.provider)

            latency = (
                observed.latency_seconds
                if (
                    observed is not None
                    and observed.latency_seconds is not None
                )
                else profile.expected_latency_seconds
            )

            if (
                maximum_latency_seconds is not None
                and latency > maximum_latency_seconds
            ):
                continue

            candidates.append(
                (profile, latency)
            )

        if not candidates:
            raise RoutingError(
                "No available provider satisfies the latency requirements."
            )

        candidates.sort(
            key=lambda item: (
                item[1],
                -item[0].priority,
            )
        )

        profile, latency = candidates[0]

        return LatencySelection(
            provider=profile.provider,
            expected_latency_seconds=latency,
            reason=(
                "Selected the available provider with the lowest "
                "expected latency."
            ),
            metadata={
                **profile.metadata,
                "candidate_count": len(candidates),
            },
        )