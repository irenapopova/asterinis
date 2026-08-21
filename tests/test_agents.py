from asterinis.agents import (
    EntityConsistencyAgent,
    ExplainabilityAgent,
    FallbackAgent,
    NLPRouterAgent,
    QueryDecompositionAgent,
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