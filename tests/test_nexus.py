import pytest

from asterinis import Nexus
from asterinis.decision import DecisionPolicy, ProviderOption
from asterinis.nlp import NLPProvider, NLPResult


class TestNLPProvider(NLPProvider):
    def analyze(self, text: str, **kwargs) -> NLPResult:
        return NLPResult(
            text=text,
            intent="classification",
        )


def test_info():
    nexus = Nexus()

    info = nexus.info()

    assert info["name"] == "Asterinis"


def test_rag_route():
    nexus = Nexus()

    result = nexus.process("Search documents about RAG.")

    assert result.route == "rag"


def test_nlp_route():
    nexus = Nexus()

    result = nexus.process("Analyze this entity using NLP.")

    assert result.route == "nlp"


def test_agent_route():
    nexus = Nexus()

    result = nexus.process("Use an agent workflow.")

    assert result.route == "agent"


def test_default_route():
    nexus = Nexus()

    result = nexus.process("Explain recursion.")

    assert result.route == "llm"


def test_empty_input():
    nexus = Nexus()

    with pytest.raises(ValueError):
        nexus.process("")


def test_nexus_unified_analyze_api() -> None:
    nexus = Nexus()
    nexus.register_nlp_provider(
        "local-nlp",
        TestNLPProvider(),
        capabilities={"classification"},
        cost=0.0,
    )

    result = nexus.analyze(
        "This is a test.",
        task="classification",
    )

    assert result.intent == "classification"
    assert result.metadata["provider"] == "local-nlp"


def test_nexus_explainable_capability_decision() -> None:
    nexus = Nexus()
    nexus.register_capability_provider(
        ProviderOption(
            name="local-retrieval",
            capability="retrieval",
            handler=lambda text: [text],
            quality=0.8,
        )
    )

    decision = nexus.decide(
        "retrieval",
        policy=DecisionPolicy(mode="local"),
    )

    assert decision.provider == "local-retrieval"
