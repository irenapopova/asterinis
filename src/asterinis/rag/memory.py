import re

from .base import Retriever
from .documents import Document, RetrievalResult


def _tokenize(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b[\w'-]+\b",
            text.lower(),
        )
    )


class InMemoryRetriever(Retriever):
    """
    Lightweight lexical retriever for small document collections.
    """

    name = "memory"

    def __init__(
        self,
        documents: list[Document] | None = None,
    ) -> None:
        self._documents = list(documents or [])

    def add(self, document: Document) -> None:
        self._documents.append(document)

    def add_many(self, documents: list[Document]) -> None:
        self._documents.extend(documents)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        query = self.validate_query(query, limit)
        query_terms = _tokenize(query)

        results: list[RetrievalResult] = []

        for document in self._documents:
            document_terms = _tokenize(document.text)

            overlap = query_terms & document_terms

            if not overlap:
                continue

            score = len(overlap) / max(len(query_terms), 1)

            results.append(
                RetrievalResult(
                    document=document,
                    score=float(score),
                    retriever=self.name,
                    metadata={
                        "matched_terms": sorted(overlap),
                    },
                )
            )

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]

    def __len__(self) -> int:
        return len(self._documents)