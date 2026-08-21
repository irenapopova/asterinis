from collections import defaultdict

from .base import Retriever
from .documents import RetrievalResult


class HybridRetriever(Retriever):
    """
    Combines results from multiple retrievers.

    Scores are combined using configurable weights.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[tuple[Retriever, float]],
    ) -> None:

        if not retrievers:
            raise ValueError(
                "At least one retriever is required."
            )

        for _, weight in retrievers:
            if weight < 0:
                raise ValueError(
                    "Retriever weights cannot be negative."
                )

        self._retrievers = retrievers

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        query = self.validate_query(query, limit)

        combined_scores: dict[str, float] = defaultdict(float)
        documents = {}
        sources: dict[str, list[str]] = defaultdict(list)

        for retriever, weight in self._retrievers:

            results = retriever.retrieve(
                query,
                limit=limit * 2,
            )

            for result in results:
                document_id = result.document.id

                documents[document_id] = result.document

                combined_scores[document_id] += (
                    result.score * weight
                )

                sources[document_id].append(
                    retriever.name
                )

        combined = [
            RetrievalResult(
                document=documents[document_id],
                score=score,
                retriever=self.name,
                metadata={
                    "sources": sources[document_id],
                },
            )
            for document_id, score
            in combined_scores.items()
        ]

        return sorted(
            combined,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]