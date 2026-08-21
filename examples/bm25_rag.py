from asterinis.rag import (
    BM25Retriever,
    Document,
)


def main() -> None:
    documents = [
        Document(
            id="1",
            text="Flair provides NLP models for named entity recognition.",
        ),
        Document(
            id="2",
            text="Asterinis provides RAG, agents, routing and orchestration.",
        ),
        Document(
            id="3",
            text="Retrieval augmented generation combines retrieval with LLMs.",
        ),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "retrieval augmented generation",
        limit=3,
    )

    for result in results:
        print(
            result.document.id,
            round(result.score, 4),
            result.document.text,
        )


if __name__ == "__main__":
    main()