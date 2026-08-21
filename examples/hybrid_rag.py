from asterinis.rag import (
    BM25Retriever,
    Document,
    FusionSource,
    ReciprocalRankFusionRetriever,
)


def main() -> None:
    documents = [
        Document(
            id="1",
            text="Flair supports named entity recognition.",
        ),
        Document(
            id="2",
            text="Asterinis combines RAG, agents and routing.",
        ),
        Document(
            id="3",
            text="Hybrid retrieval combines multiple retrieval strategies.",
        ),
    ]

    first = BM25Retriever(documents)
    second = BM25Retriever(documents)

    retriever = ReciprocalRankFusionRetriever(
        [
            FusionSource(first, weight=1.0),
            FusionSource(second, weight=1.0),
        ]
    )

    results = retriever.retrieve(
        "hybrid retrieval",
        limit=3,
    )

    for result in results:
        print(
            result.document.id,
            round(result.score, 4),
            result.metadata,
        )


if __name__ == "__main__":
    main()