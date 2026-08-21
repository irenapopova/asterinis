from __future__ import annotations

from typing import Any

from .config import AsterinisConfig
from .registry import ProviderRegistry
from .result import NexusResult
from .router import Router as BaseRouter


class Router(BaseRouter):
    """Backward-compatible router API formerly defined in this module."""

    def __init__(self, default_route: str = "llm") -> None:
        super().__init__(default_route=default_route)
        self.add_route(
            "rag",
            lambda text: any(
                word in text.lower()
                for word in ("document", "source", "retrieve", "search")
            ),
        )
        self.add_route(
            "nlp",
            lambda text: any(
                word in text.lower()
                for word in ("entity", "ner", "language", "nlp")
            ),
        )
        self.add_route(
            "agent",
            lambda text: any(
                word in text.lower()
                for word in ("agent", "tool", "workflow")
            ),
        )

    def add_route(
        self,
        name: str,
        rule: Any,
        *,
        first: bool = False,
    ) -> None:
        self.register(name, rule, priority=1 if first else 0)

    def route(self, text: str) -> str:
        return self.resolve(text)


class Nexus:
    """Main orchestration interface for Asterinis."""

    def __init__(
        self,
        config: AsterinisConfig | None = None,
    ) -> None:
        self.config = config or AsterinisConfig()
        self.router = Router(default_route=self.config.default_route)
        self.providers = ProviderRegistry()

    def info(self) -> dict[str, str]:
        return {"name": "Asterinis"}

    def register_provider(
        self,
        route: str,
        provider: Any,
        *,
        priority: int = 0,
        predicate: Any = None,
    ) -> None:
        self.providers.register(route, provider)

        if predicate is not None:
            self.router.register(route, predicate, priority=priority)

    def process(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> NexusResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")

        route = self.router.resolve(text)
        request_metadata = metadata or {}

        if not self.providers.contains(route):
            return NexusResult(
                route=route,
                provider=None,
                output=None,
                metadata={
                    **request_metadata,
                    "message": (
                        f"No provider registered for route '{route}'."
                    ),
                },
            )

        provider = self.providers.get(route)
        output = provider.invoke(
            text,
            metadata=request_metadata,
        )

        return NexusResult(
            route=route,
            provider=provider.name,
            output=output,
            metadata=request_metadata,
        )
