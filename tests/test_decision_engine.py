import pytest

from asterinis.decision import DecisionEngine, DecisionPolicy, ProviderOption
from asterinis.exceptions import RoutingError


def test_decision_engine_explains_selection() -> None:
    engine = DecisionEngine()
    engine.register(
        ProviderOption(
            name="local-rag",
            capability="retrieval",
            handler=lambda text: [text],
            quality=0.8,
            cost=0.0,
            latency_ms=20,
        )
    )
    engine.register(
        ProviderOption(
            name="openai",
            capability="retrieval",
            handler=lambda text: [text],
            quality=0.95,
            cost=0.02,
            latency_ms=200,
            external=True,
        )
    )

    decision = engine.decide("retrieval", policy=DecisionPolicy(mode="local"))

    assert decision.provider == "local-rag"
    assert "local-rag" in decision.reason


def test_decision_engine_enforces_budget_and_privacy() -> None:
    engine = DecisionEngine()
    engine.register(
        ProviderOption(
            name="external-llm",
            capability="generation",
            handler=lambda text: text,
            quality=1.0,
            cost=0.10,
            external=True,
        )
    )

    with pytest.raises(RoutingError):
        engine.decide(
            "generation",
            policy=DecisionPolicy(max_cost=0.01, sensitive=True),
        )


def test_decision_engine_records_performance() -> None:
    engine = DecisionEngine()
    engine.register(
        ProviderOption(
            name="local-classifier",
            capability="classification",
            handler=lambda text: "positive",
        )
    )

    decision, output = engine.execute("great", "classification")

    assert decision.provider == "local-classifier"
    assert output == "positive"
    assert len(engine.learning_store) == 1
