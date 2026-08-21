from collections.abc import Callable
from math import sqrt

from .base import Retriever
from .documents import Document, RetrievalResult


Embedding = list[float]
Embedder = Callable[[str], Embedding]


def cosine_similarity(
    left: Embedding,
    right: Embedding,
) -> float:

    if len(left) != len(right):
        raise ValueError(
            "Embedding dimensions must match."
        )

    dot = sum(a * b for a, b in zip(left, right))

    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


class VectorRetriever(Retriever):
    """
    Semantic retriever using user-provided embedding functions.
    """

    name = "vector"

    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder,
    ) -> None:

        if not callable(embedder):
            raise TypeError("embedder must be callable.")

        self._documents = documents
        self._embedder = embedder

        self._embeddings = {
            document.id: self._embedder(document.text)
            for document in documents
        }

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        query = self.validate_query(query, limit)

        query_embedding = self._embedder(query)

        results: list[RetrievalResult] = []

        for document in self._documents:
            score = cosine_similarity(
                query_embedding,
                self._embeddings[document.id],
            )

            results.append(
                RetrievalResult(
                    document=document,
                    score=score,
                    retriever=self.name,
                )
            )

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]