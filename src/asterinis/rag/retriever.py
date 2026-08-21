from .base import Retriever
from .documents import Document


class InMemoryRetriever(Retriever):
    """
    Lightweight in-memory retriever.

    Intended for testing, small datasets, and early development.
    """

    name = "memory"

    def __init__(self, documents: list[Document] | None = None):
        self._documents = documents or []

    def add(self, document: Document) -> None:
        self._documents.append(document)

    def add_many(self, documents: list[Document]) -> None:
        self._documents.extend(documents)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[Document]:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip().lower()

        if not query:
            raise ValueError("Query cannot be empty.")

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        query_terms = set(query.split())

        scored: list[tuple[int, Document]] = []

        for document in self._documents:
            document_terms = set(document.text.lower().split())

            score = len(query_terms.intersection(document_terms))

            if score > 0:
                scored.append((score, document))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            document
            for _, document in scored[:limit]
        ]

    def __len__(self) -> int:
        return len(self._documents)