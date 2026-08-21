import pytest

from asterinis.agents import (
    EntityConsistencyAgent,
    ContradictionAgent,
    ExplainabilityAgent,
    FallbackAgent,
    NLPRouterAgent,
    QueryDecompositionAgent,
    QueryPlanStep,
    QueryPlannerAgent,
    VerificationAgent,
)


def test_query_decomposition() -> None:
    agent = QueryDecompositionAgent(
        lambda query: [
            "What is Flair?",
            "How does Asterinis use Flair?",
        ]
    )

    result = agent.run(
        "What is Flair and how does Asterinis use it?"
    )

    assert result.count == 2


def test_entity_consistency_full_match() -> None:
    agent = EntityConsistencyAgent()

    result = agent.evaluate(
        ["Deutsche Bank", "Frankfurt"],
        ["Frankfurt", "Deutsche Bank", "Germany"],
    )

    assert result.consistent
    assert result.score == 1.0
    assert result.missing_entities == []


def test_entity_consistency_detects_missing_entity() -> None:
    agent = EntityConsistencyAgent(
        consistency_threshold=1.0
    )

    result = agent.evaluate(
        ["Deutsche Bank", "Frankfurt"],
        ["Deutsche Bank"],
    )

    assert not result.consistent
    assert result.score == 0.5
    assert "frankfurt" in result.missing_entities


def test_nlp_router_uses_metadata() -> None:
    router = NLPRouterAgent(
        default_route="llm"
    )

    router.register(
        "rag",
        lambda text, metadata: bool(
            metadata.get("entities")
        ),
        reason="Entities are available for retrieval.",
        priority=10,
    )

    decision = router.decide(
        "Tell me about Berlin.",
        nlp_metadata={
            "entities": ["Berlin"],
            "confidence": 0.94,
        },
    )

    assert decision.route == "rag"
    assert decision.confidence == 0.94


def test_verification_agent() -> None:
    agent = VerificationAgent(
        lambda value, metadata: value == metadata["expected"]
    )

    result = agent.verify(
        "Berlin",
        metadata={
            "expected": "Berlin",
        },
    )

    assert result.verified


def test_fallback_agent() -> None:
    agent = FallbackAgent(
        lambda reason, metadata: (
            f"Fallback used because: {reason}"
        )
    )

    result = agent.run(
        "Primary provider unavailable."
    )

    assert result.succeeded
    assert "Primary provider unavailable" in result.output


def test_explainability_agent() -> None:
    agent = ExplainabilityAgent()

    explanation = agent.explain(
        route="rag",
        agent="retrieval-quality",
        confidence=0.91,
        evidence_count=5,
        decision="generate",
        reasons=[
            "Retrieved evidence passed the quality threshold."
        ],
    )

    assert explanation.route == "rag"
    assert explanation.decision == "generate"
    assert explanation.confidence == 0.91


def test_query_planner_builds_valid_plan() -> None:
    def planner(query: str, context: dict) -> list[QueryPlanStep]:
        return [
            QueryPlanStep(
                action="retrieve",
                input=query,
                reason="Retrieve evidence.",
            )
        ]

    agent = QueryPlannerAgent(planner)
    plan = agent.plan("Test query")

    assert plan.query == "Test query"
    assert plan.step_count == 1
    assert plan.steps[0].action == "retrieve"


def test_query_planner_enforces_max_steps() -> None:
    def planner(query: str, context: dict) -> list[QueryPlanStep]:
        return [
            QueryPlanStep(
                action="retrieve",
                input=query,
                reason="Retrieve evidence.",
            ),
            QueryPlanStep(
                action="verify",
                input=query,
                reason="Verify evidence.",
            ),
        ]

    agent = QueryPlannerAgent(planner, max_steps=1)

    with pytest.raises(ValueError):
        agent.plan("Test query")


def test_query_planner_rejects_disallowed_action() -> None:
    def planner(query: str, context: dict) -> list[QueryPlanStep]:
        return [
            QueryPlanStep(
                action="delete_everything",
                input=query,
                reason="Invalid test action.",
            )
        ]

    agent = QueryPlannerAgent(planner)

    with pytest.raises(ValueError):
        agent.plan("Test query")


def test_contradiction_agent_detects_conflict() -> None:
    def comparator(first: str, second: str) -> float:
        if "Berlin" in first and "Frankfurt" in second:
            return 0.95
        if "Frankfurt" in first and "Berlin" in second:
            return 0.95
        return 0.10

    agent = ContradictionAgent(comparator, threshold=0.80)
    result = agent.evaluate(
        [
            "The company is based in Berlin.",
            "The company is based in Frankfurt.",
        ]
    )

    assert result.detected
    assert result.severity == 0.95
    assert len(result.pairs) == 1


def test_contradiction_agent_reports_no_conflict() -> None:
    agent = ContradictionAgent(
        lambda first, second: 0.10,
        threshold=0.80,
    )
    result = agent.evaluate(
        [
            "Asterinis supports RAG.",
            "Asterinis supports agents.",
        ]
    )

    assert not result.detected
    assert result.severity == 0.0
    assert result.pairs == []


def test_contradiction_agent_rejects_invalid_score() -> None:
    agent = ContradictionAgent(lambda first, second: 1.5)

    with pytest.raises(ValueError):
        agent.evaluate(
            [
                "First statement.",
                "Second statement.",
            ]
        )
