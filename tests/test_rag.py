import pytest

from asterinis.rag import (
    ConsensusRetriever,
    ConsensusSource,
)
from asterinis.rag.bm25 import BM25Retriever
from asterinis.rag.documents import Document
from asterinis.rag.fusion import (
    FusionSource,
    ReciprocalRankFusionRetriever,
)


def sample_documents() -> list[Document]:
    return [
        Document(
            id="flair",
            text=(
                "Flair is a framework for natural language "
                "processing and named entity recognition."
            ),
        ),
        Document(
            id="rag",
            text=(
                "Retrieval augmented generation uses retrieved "
                "documents to provide context to language models."
            ),
        ),
        Document(
            id="asterinis",
            text=(
                "Asterinis combines retrieval, NLP, routing, "
                "agents and AI orchestration."
            ),
        ),
    ]


def test_bm25_returns_relevant_document() -> None:
    retriever = BM25Retriever(sample_documents())

    results = retriever.retrieve(
        "named entity recognition",
        limit=3,
    )

    assert results
    assert results[0].document.id == "flair"
    assert results[0].score > 0


def test_bm25_returns_empty_for_unmatched_query() -> None:
    retriever = BM25Retriever(sample_documents())

    results = retriever.retrieve(
        "quantum superconductivity",
    )

    assert results == []


def test_fusion_combines_retrievers() -> None:
    documents = sample_documents()

    first = BM25Retriever(documents)
    second = BM25Retriever(documents)

    fusion = ReciprocalRankFusionRetriever(
        [
            FusionSource(first),
            FusionSource(second),
        ]
    )

    results = fusion.retrieve(
        "retrieval language models",
        limit=2,
    )

    assert results
    assert len(results) <= 2
    assert results[0].retriever == "reciprocal-rank-fusion"
    assert results[0].metadata["sources"]


def test_consensus_retriever_rewards_agreement() -> None:
    documents = sample_documents()

    first = BM25Retriever(documents)
    second = BM25Retriever(documents)

    retriever = ConsensusRetriever(
        [
            ConsensusSource(first),
            ConsensusSource(second),
        ]
    )

    results = retriever.retrieve(
        "named entity recognition",
        limit=3,
    )

    assert results
    assert results[0].document.id == "flair"
    assert results[0].retriever == "consensus"
    assert results[0].metadata["agreement"] == 1.0


def test_consensus_retriever_respects_minimum_agreement() -> None:
    documents = sample_documents()

    first = BM25Retriever(documents)
    second = BM25Retriever(documents)

    retriever = ConsensusRetriever(
        [
            ConsensusSource(first),
            ConsensusSource(second),
        ],
        minimum_agreement=1.0,
    )

    results = retriever.retrieve(
        "retrieval language models",
        limit=3,
    )

    assert all(
        result.metadata["agreement"] == 1.0
        for result in results
    )


def test_consensus_source_rejects_invalid_weight() -> None:
    retriever = BM25Retriever(sample_documents())

    with pytest.raises(ValueError):
        ConsensusSource(retriever, weight=0)
