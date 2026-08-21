from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from asterinis.exceptions import RoutingError


@dataclass(slots=True)
class CapabilityProfile:
    """
    Describes the capabilities exposed by a provider.

    Capabilities are represented as simple strings so applications can define
    their own vocabulary without coupling Asterinis to a fixed provider model.
    """

    provider: str
    capabilities: set[str] = field(default_factory=set)
    priority: int = 0

    def __post_init__(self) -> None:
        self.provider = self.provider.strip()

        if not self.provider:
            raise ValueError("provider cannot be empty.")

        self.capabilities = {
            capability.strip()
            for capability in self.capabilities
            if capability.strip()
        }

    def supports(self, capability: str) -> bool:
        capability = capability.strip()

        if not capability:
            raise ValueError("capability cannot be empty.")

        return capability in self.capabilities


class CapabilityRouter:
    """
    Selects providers according to the capabilities required by a request.

    A request may require one or several capabilities. Providers can be
    selected using either strict matching ("all") or partial matching ("any").
    """

    def __init__(self) -> None:
        self._profiles: dict[str, CapabilityProfile] = {}

    def register(
        self,
        profile: CapabilityProfile,
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

    def remove(self, provider: str) -> None:
        self._profiles.pop(provider, None)

    def select(
        self,
        required: Iterable[str],
        *,
        mode: str = "all",
    ) -> CapabilityProfile:
        required_capabilities = {
            capability.strip()
            for capability in required
            if capability.strip()
        }

        if not required_capabilities:
            raise ValueError(
                "At least one required capability must be provided."
            )

        if mode not in {"all", "any"}:
            raise ValueError(
                "mode must be either 'all' or 'any'."
            )

        candidates: list[CapabilityProfile] = []

        for profile in self._profiles.values():
            if mode == "all":
                matches = required_capabilities.issubset(
                    profile.capabilities
                )
            else:
                matches = bool(
                    required_capabilities
                    & profile.capabilities
                )

            if matches:
                candidates.append(profile)

        if not candidates:
            raise RoutingError(
                "No registered provider satisfies the requested capabilities."
            )

        candidates.sort(
            key=lambda profile: profile.priority,
            reverse=True,
        )

        return candidates[0]

    def providers_for(
        self,
        capability: str,
    ) -> tuple[str, ...]:
        capability = capability.strip()

        if not capability:
            raise ValueError("capability cannot be empty.")

        matches = [
            profile
            for profile in self._profiles.values()
            if capability in profile.capabilities
        ]

        matches.sort(
            key=lambda profile: profile.priority,
            reverse=True,
        )

        return tuple(
            profile.provider
            for profile in matches
        )