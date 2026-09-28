import pytest

from asterinis.exceptions import RoutingError
from asterinis.nlp import (
    NLPProvider,
    NLPProviderProfile,
    NLPProviderRegistry,
    NLPResult,
    NLPTaskRouter,
)
from asterinis.nlp.providers import TextClassifierProvider
from asterinis.nlp.classification import TextClassifier


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


def test_text_classifier_provider_adapts_to_nlp_result() -> None:
    classifier = TextClassifier(
        lambda text: [
            ("positive", 0.9),
            ("negative", 0.1),
        ]
    )
    provider = TextClassifierProvider(
        classifier,
        name="local-classifier",
    )

    result = provider.analyze("This is excellent.")

    assert result.classifications[0].label == "positive"
    assert result.confidence == 0.5
    assert result.metadata["provider"] == "local-classifier"


def test_text_classifier_provider_works_with_task_router() -> None:
    provider = TextClassifierProvider(
        TextClassifier(lambda text: [("positive", 0.95)]),
        name="local-classifier",
    )
    router = NLPTaskRouter()
    router.register(
        "local-classifier",
        provider,
        capabilities={"classification"},
        cost=0.0,
    )

    result = router.analyze(
        "This is excellent.",
        task="classification",
    )

    assert result.classifications[0].label == "positive"
    assert result.metadata["provider"] == "local-classifier"


def test_nlp_router_learns_from_provider_feedback() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
        quality=0.80,
    )
    router.register(
        "flair",
        StubProvider("flair"),
        capabilities={"classification"},
        quality=0.90,
    )

    for _ in range(3):
        router.record_feedback(
            "local",
            "classification",
            success=True,
            quality=1.0,
            confidence=1.0,
            latency_ms=5.0,
            cost=0.0,
        )
        router.record_feedback(
            "flair",
            "classification",
            success=False,
            quality=0.0,
            confidence=0.0,
            latency_ms=100.0,
            cost=0.01,
        )

    selection = router.select("classification")

    assert selection.provider == "local"
    assert selection.metadata["learning_samples"] == 3


def test_nlp_router_records_feedback_after_analysis() -> None:
    router = NLPTaskRouter()
    router.register(
        "local",
        StubProvider("local"),
        capabilities={"classification"},
    )

    router.analyze("A simple sentence.", task="classification")

    records = router.learning_store.for_strategy("local")
    assert len(records) == 1
    assert records[0].query_type == "classification"
    assert records[0].success
