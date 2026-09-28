from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NLPProviderProfile:
    """Capabilities and operating characteristics of an NLP provider."""

    name: str
    provider: Any
    capabilities: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    quality: float = 0.5
    cost: float = 0.0
    latency_ms: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("provider name cannot be empty.")

        self.capabilities = {
            value.strip().lower()
            for value in self.capabilities
            if value.strip()
        }
        self.languages = {
            value.strip().lower()
            for value in self.languages
            if value.strip()
        }

        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1.")
        if self.cost < 0:
            raise ValueError("cost cannot be negative.")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative.")


class NLPProviderRegistry:
    """Registry of NLP providers available to the task router."""

    def __init__(self) -> None:
        self._profiles: dict[str, NLPProviderProfile] = {}

    def register(
        self,
        profile: NLPProviderProfile,
        *,
        replace: bool = False,
    ) -> None:
        if profile.name in self._profiles and not replace:
            raise ValueError(
                f"Provider '{profile.name}' is already registered."
            )
        self._profiles[profile.name] = profile

    def get(self, name: str) -> NLPProviderProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Provider '{name}' is not registered.") from exc

    def remove(self, name: str) -> None:
        self._profiles.pop(name, None)

    def profiles(self) -> tuple[NLPProviderProfile, ...]:
        return tuple(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)
