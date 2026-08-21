import pytest

from asterinis import Nexus


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