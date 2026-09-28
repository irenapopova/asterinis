from __future__ import annotations

from typing import Any

from .config import AsterinisConfig
from .async_providers import invoke_provider
from .decision import Decision, DecisionEngine, DecisionPolicy, ProviderOption
from .nlp.fallback import NLPFallbackRouter
from .nlp.router import NLPTaskRouter
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
        self.nlp_router = NLPTaskRouter()
        self.nlp_fallback = NLPFallbackRouter(self.nlp_router)
        self.decision_engine = DecisionEngine()

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

    def register_nlp_provider(
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
        """Register an NLP provider for the unified analyze API."""
        self.nlp_router.register(
            name,
            provider,
            capabilities=capabilities,
            languages=languages,
            quality=quality,
            cost=cost,
            latency_ms=latency_ms,
            replace=replace,
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
        """Analyze text through the learned NLP router and fallbacks."""
        return self.nlp_fallback.analyze(
            text,
            task=task,
            language=language,
            max_cost=max_cost,
            max_latency_ms=max_latency_ms,
            **kwargs,
        )

    def register_capability_provider(
        self,
        option: ProviderOption,
        *,
        replace: bool = False,
    ) -> None:
        """Register an NLP, retrieval, agent, or LLM decision option."""
        self.decision_engine.register(option, replace=replace)

    def decide(
        self,
        capability: str,
        *,
        policy: DecisionPolicy | None = None,
    ) -> Decision:
        """Return an explainable provider decision."""
        return self.decision_engine.decide(capability, policy=policy)

    def execute_capability(
        self,
        text: str,
        capability: str,
        *,
        policy: DecisionPolicy | None = None,
    ) -> tuple[Decision, Any]:
        """Select and execute a registered capability provider."""
        return self.decision_engine.execute(
            text,
            capability,
            policy=policy,
        )

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

    async def process_async(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
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
        output = await invoke_provider(
            provider,
            text,
            timeout_seconds=timeout_seconds,
            metadata=request_metadata,
        )

        return NexusResult(
            route=route,
            provider=provider.name,
            output=output,
            metadata=request_metadata,
        )
