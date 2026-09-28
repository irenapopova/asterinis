import pytest

from asterinis.exceptions import RoutingError
from asterinis.nlp import (
    NLPProvider,
    NLPProviderProfile,
    NLPProviderRegistry,
    NLPResult,
    NLPTaskRouter,
    NLPFallbackRouter,
)
from asterinis.nlp.providers import TextClassifierProvider
from asterinis.nlp.classification import TextClassifier
from asterinis.nlp.embeddings import HashEmbeddingProvider
from asterinis.nlp.providers import (
    EmbeddingNLPProvider,
    FastTextLanguageProvider,
    LanguageNLPProvider,
    SentenceTransformerEmbeddingProvider,
    SentimentNLPProvider,
    TransformersSentimentProvider,
)
from asterinis.nlp.language import LanguageDetector
from asterinis.nlp.evaluation import evaluate_labels
from asterinis.nlp.models import ModelCard, NLPModelRegistry


class StubProvider(NLPProvider):
    def __init__(self, label: str) -> None:
        self.label = label

    def analyze(self, text: str, **kwargs) -> NLPResult:
        return NLPResult(
            text=text,
            intent=self.label,
        )


class FailingProvider(NLPProvider):
    def analyze(self, text: str, **kwargs) -> NLPResult:
        raise RuntimeError("provider unavailable")


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


def test_nlp_fallback_router_uses_next_provider() -> None:
    router = NLPTaskRouter()
    router.register(
        "primary",
        FailingProvider(),
        capabilities={"classification"},
        quality=0.99,
    )
    router.register(
        "fallback",
        StubProvider("fallback"),
        capabilities={"classification"},
        quality=0.80,
    )

    result = NLPFallbackRouter(router).analyze(
        "A simple sentence.",
        task="classification",
    )

    assert result.intent == "fallback"
    assert result.metadata["provider"] == "fallback"
    assert result.metadata["fallback_attempts"] == 2
    assert result.metadata["fallback_failures"] == [
        "primary: RuntimeError"
    ]


def test_nlp_fallback_router_respects_max_attempts() -> None:
    router = NLPTaskRouter()
    router.register(
        "primary",
        FailingProvider(),
        capabilities={"classification"},
        quality=0.99,
    )
    router.register(
        "fallback",
        StubProvider("fallback"),
        capabilities={"classification"},
        quality=0.80,
    )

    with pytest.raises(RoutingError):
        NLPFallbackRouter(router, max_attempts=1).analyze(
            "A simple sentence.",
            task="classification",
        )


def test_local_language_provider() -> None:
    provider = LanguageNLPProvider(
        LanguageDetector(lambda text: ("en", 0.99))
    )
    result = provider.analyze("Hello world")
    assert result.language == "en"
    assert result.confidence == 0.99


def test_local_sentiment_provider() -> None:
    result = SentimentNLPProvider().analyze("This is excellent.")
    assert result.intent == "positive"
    assert result.metadata["sentiment"] == "positive"


def test_local_embedding_provider() -> None:
    provider = EmbeddingNLPProvider(HashEmbeddingProvider(dimensions=8))
    result = provider.analyze("retrieval")
    assert len(result.metadata["embedding"]) == 8
    assert result.metadata["dimensions"] == 8


def test_sentence_transformer_provider_with_injected_model() -> None:
    class FakeModel:
        def encode(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    provider = SentenceTransformerEmbeddingProvider(model=FakeModel())
    result = provider.analyze("semantic text")

    assert result.metadata["embedding"] == [0.1, 0.2, 0.3]
    assert result.metadata["dimensions"] == 3


def test_transformers_sentiment_provider_with_injected_pipeline() -> None:
    provider = TransformersSentimentProvider(
        pipeline=lambda text: [{"label": "POSITIVE", "score": 0.97}]
    )
    result = provider.analyze("Great result")

    assert result.intent == "positive"
    assert result.confidence == 0.97


def test_fasttext_language_provider_with_injected_model() -> None:
    class FakeModel:
        def predict(self, text: str, k: int) -> tuple[list[str], list[float]]:
            return ["__label__en"], [0.99]

    provider = FastTextLanguageProvider(model=FakeModel())
    result = provider.analyze("Hello world")

    assert result.language == "en"
    assert result.confidence == 0.99


def test_model_registry_loads_models_lazily() -> None:
    registry = NLPModelRegistry()
    created = []

    def factory() -> NLPProvider:
        created.append(True)
        return StubProvider("local")

    registry.register(
        ModelCard(name="local-model", task="classification"),
        factory,
    )

    assert created == []
    assert registry.load("local-model") is registry.load("local-model")
    assert len(created) == 1


def test_label_evaluation() -> None:
    evaluation = evaluate_labels(
        ["positive", "negative", "neutral"],
        ["positive", "negative", "positive"],
    )
    assert evaluation.total == 3
    assert evaluation.correct == 2
    assert evaluation.accuracy == 2 / 3
