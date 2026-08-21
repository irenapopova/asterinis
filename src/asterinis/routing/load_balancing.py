from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from asterinis.exceptions import RoutingError

from .health import HealthRegistry


@dataclass(slots=True)
class LoadBalancedProvider:
    provider: str
    weight: int = 1

    def __post_init__(self) -> None:
        self.provider = self.provider.strip()

        if not self.provider:
            raise ValueError("provider cannot be empty.")

        if self.weight < 1:
            raise ValueError(
                "weight must be greater than zero."
            )


class RoundRobinRouter:
    """
    Distributes requests across available providers using round-robin routing.

    Unavailable providers are skipped automatically when a HealthRegistry is
    supplied.
    """

    def __init__(
        self,
        providers: list[LoadBalancedProvider],
        *,
        health: HealthRegistry | None = None,
    ) -> None:
        if not providers:
            raise ValueError(
                "At least one provider is required."
            )

        self.health = health or HealthRegistry()
        self._providers = list(providers)
        self._index = 0
        self._lock = Lock()

    def select(self) -> str:
        with self._lock:
            provider_count = len(self._providers)

            for _ in range(provider_count):
                profile = self._providers[
                    self._index % provider_count
                ]

                self._index = (
                    self._index + 1
                ) % provider_count

                if self.health.available(
                    profile.provider
                ):
                    return profile.provider

        raise RoutingError(
            "No load-balanced provider is currently available."
        )


class WeightedRoundRobinRouter:
    """
    Distributes requests according to provider weights.

    A provider with weight 3 receives roughly three selections for every one
    selection of a provider with weight 1, while unavailable providers are
    skipped.
    """

    def __init__(
        self,
        providers: list[LoadBalancedProvider],
        *,
        health: HealthRegistry | None = None,
    ) -> None:
        if not providers:
            raise ValueError(
                "At least one provider is required."
            )

        self.health = health or HealthRegistry()
        self._schedule = self._build_schedule(
            providers
        )
        self._index = 0
        self._lock = Lock()

    @staticmethod
    def _build_schedule(
        providers: list[LoadBalancedProvider],
    ) -> list[str]:
        schedule: list[str] = []

        for provider in providers:
            schedule.extend(
                [provider.provider]
                * provider.weight
            )

        return schedule

    def select(self) -> str:
        with self._lock:
            schedule_size = len(self._schedule)

            for _ in range(schedule_size):
                provider = self._schedule[
                    self._index % schedule_size
                ]

                self._index = (
                    self._index + 1
                ) % schedule_size

                if self.health.available(provider):
                    return provider

        raise RoutingError(
            "No weighted provider is currently available."
        )