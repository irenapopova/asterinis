from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import monotonic


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class ProviderHealth:
    provider: str
    status: HealthStatus
    latency_seconds: float | None = None
    checked_at: float = 0.0

    @property
    def available(self) -> bool:
        return self.status is not HealthStatus.UNAVAILABLE


class HealthRegistry:
    """Keeps the latest known health state for providers."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderHealth] = {}

    def update(
        self,
        provider: str,
        status: HealthStatus,
        *,
        latency_seconds: float | None = None,
    ) -> ProviderHealth:
        provider = provider.strip()

        if not provider:
            raise ValueError(
                "provider cannot be empty."
            )

        if (
            latency_seconds is not None
            and latency_seconds < 0
        ):
            raise ValueError(
                "latency_seconds cannot be negative."
            )

        health = ProviderHealth(
            provider=provider,
            status=status,
            latency_seconds=latency_seconds,
            checked_at=monotonic(),
        )

        self._providers[provider] = health

        return health

    def get(
        self,
        provider: str,
    ) -> ProviderHealth | None:
        return self._providers.get(provider)

    def available(
        self,
        provider: str,
    ) -> bool:
        health = self.get(provider)

        return (
            health is None
            or health.available
        )