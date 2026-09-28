import pytest

from asterinis.exceptions import RoutingError
from asterinis.nlp import (
    NLPProvider,
    NLPProviderProfile,
    NLPProviderRegistry,
    NLPResult,
    NLPTaskRouter,
)


class StubProvider(NLPProvider):
    def __init__(self, label: str) -> None:
        self.label = label

    def analyze(self, text: str, **kwargs) -> NLPResult:
        return NLPResult(
            text=text,
            intent=self.label,
        )


def test_nlp_registry_registers_and_retrieves_provider() -> None:
    registry = NLPProviderRegistry()
    profile = NLPProviderProfile(
        name="local",
        provider=StubProvider("local"),
        capabilities={"classification"},
    )

    registry.register(profile)

    assert registry.get("local") is profile
    assert len(registry) == 1


def test_nlp_router_selects_highest_quality_provider() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
        quality=0.80,
        cost=0.0,
        latency_ms=20.0,
    )
    router.register(
        "flair",
        StubProvider("flair"),
        capabilities={"classification"},
        quality=0.95,
        cost=0.01,
        latency_ms=40.0,
    )

    selection = router.select("classification")

    assert selection.provider == "flair"
    assert selection.quality == 0.95


def test_nlp_router_respects_cost_constraint() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
        quality=0.80,
        cost=0.0,
    )
    router.register(
        "paid",
        StubProvider("paid"),
        capabilities={"classification"},
        quality=0.99,
        cost=0.10,
    )

    selection = router.select("classification", max_cost=0.01)

    assert selection.provider == "local"


def test_nlp_router_analyzes_with_selected_provider() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
    )

    result = router.analyze(
        "A simple sentence.",
        task="classification",
    )

    assert result.intent == "local"
    assert result.metadata["provider"] == "local"
    assert result.metadata["task"] == "classification"


def test_nlp_router_rejects_unsupported_task() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
    )

    with pytest.raises(RoutingError):
        router.select("ner")
