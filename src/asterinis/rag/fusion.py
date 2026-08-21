from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .base import Retriever
from .documents import Document, RetrievalResult


@dataclass(slots=True)
class FusionSource:
    retriever: Retriever
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(
                "Retriever weight must be greater than zero."
            )


class ReciprocalRankFusionRetriever(Retriever):
    """
    Combines ranked results from multiple retrievers using Reciprocal
    Rank Fusion (RRF).

    RRF works with rank positions rather than assuming that score values
    produced by different retrieval systems are directly comparable.
    """

    name = "reciprocal-rank-fusion"

    def __init__(
        self,
        sources: list[FusionSource],
        *,
        rank_constant: int = 60,
        candidate_limit: int = 20,
    ) -> None:
        if not sources:
            raise ValueError(
                "At least one retrieval source is required."
            )

        if rank_constant < 1:
            raise ValueError(
                "rank_constant must be greater than zero."
            )

        if candidate_limit < 1:
            raise ValueError(
                "candidate_limit must be greater than zero."
            )

        self.sources = list(sources)
        self.rank_constant = rank_constant
        self.candidate_limit = candidate_limit

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query = self.validate_query(query, limit)

        fused_scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        source_metadata: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for source in self.sources:
            results = source.retriever.retrieve(
                query,
                limit=self.candidate_limit,
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                document_id = result.document.id

                fused_scores[document_id] += (
                    source.weight
                    / (self.rank_constant + rank)
                )

                documents[document_id] = result.document

                source_metadata[document_id].append(
                    {
                        "retriever": result.retriever,
                        "rank": rank,
                        "score": result.score,
                        "weight": source.weight,
                    }
                )

        ranked_document_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        return [
            RetrievalResult(
                document=documents[document_id],
                score=fused_scores[document_id],
                retriever=self.name,
                metadata={
                    "sources": source_metadata[document_id],
                    "rank_constant": self.rank_constant,
                },
            )
            for document_id in ranked_document_ids[:limit]
        ]