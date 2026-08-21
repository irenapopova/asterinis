from asterinis.rag import (
    Document,
    EntityAwareRetriever,
    InMemoryRetriever,
)


def extract_entities(text: str) -> list[str]:
    known_entities = [
        "Berlin",
        "Flair",
        "Asterinis",
    ]

    return [
        entity
        for entity in known_entities
        if entity.lower() in text.lower()
    ]


def main() -> None:
    documents = [
        Document(
            id="1",
            text="Flair is an NLP framework.",
        ),
        Document(
            id="2",
            text="Asterinis uses NLP signals inside RAG workflows.",
        ),
        Document(
            id="3",
            text="Berlin is a major technology and research hub.",
        ),
    ]

    base_retriever = InMemoryRetriever(documents)

    retriever = EntityAwareRetriever(
        base_retriever,
        entity_extractor=extract_entities,
    )

    results = retriever.retrieve(
        "How can Flair support Asterinis?"
    )

    for result in results:
        print(
            result.document.id,
            round(result.score, 4),
            result.document.text,
        )


if __name__ == "__main__":
    main()