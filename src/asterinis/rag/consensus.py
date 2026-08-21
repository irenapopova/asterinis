from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .base import Retriever
from .documents import Document, RetrievalResult


@dataclass(slots=True)
class RetrieverVote:
    retriever: str
    rank: int
    score: float
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": self.retriever,
            "rank": self.rank,
            "score": self.score,
            "weight": self.weight,
        }


@dataclass(slots=True)
class ConsensusResult:
    document: Document
    agreement: float
    votes: list[RetrieverVote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retriever_count(self) -> int:
        return len(self.votes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": {
                "id": self.document.id,
                "text": self.document.text,
                "metadata": dict(self.document.metadata),
            },
            "agreement": self.agreement,
            "retriever_count": self.retriever_count,
            "votes": [vote.to_dict() for vote in self.votes],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ConsensusSource:
    retriever: Retriever
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Source weight must be greater than zero.")


class ConsensusRetriever(Retriever):
    """
    Combines results from multiple retrievers and measures how strongly
    they agree on the same documents.

    Agreement is based on weighted retriever participation rather than raw
    retrieval scores, because scores from different retrieval systems are
    not always directly comparable.
    """

    name = "consensus"

    def __init__(
        self,
        sources: list[ConsensusSource],
        *,
        candidate_limit: int = 20,
        minimum_agreement: float = 0.0,
    ) -> None:
        if not sources:
            raise ValueError("At least one consensus source is required.")

        if candidate_limit < 1:
            raise ValueError("candidate_limit must be greater than zero.")

        if not 0.0 <= minimum_agreement <= 1.0:
            raise ValueError(
                "minimum_agreement must be between 0 and 1."
            )

        self.sources = list(sources)
        self.candidate_limit = candidate_limit
        self.minimum_agreement = minimum_agreement
        self._total_weight = sum(source.weight for source in self.sources)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query = self.validate_query(query, limit)

        documents: dict[str, Document] = {}
        votes: dict[str, list[RetrieverVote]] = defaultdict(list)
        weighted_support: dict[str, float] = defaultdict(float)

        for source in self.sources:
            results = source.retriever.retrieve(
                query,
                limit=self.candidate_limit,
            )

            seen_in_source: set[str] = set()

            for rank, result in enumerate(results, start=1):
                document_id = result.document.id

                if document_id in seen_in_source:
                    continue

                seen_in_source.add(document_id)
                documents[document_id] = result.document

                votes[document_id].append(
                    RetrieverVote(
                        retriever=result.retriever,
                        rank=rank,
                        score=float(result.score),
                        weight=source.weight,
                    )
                )

                weighted_support[document_id] += source.weight

        consensus_results: list[ConsensusResult] = []

        for document_id, document in documents.items():
            agreement = (
                weighted_support[document_id] / self._total_weight
                if self._total_weight
                else 0.0
            )

            if agreement < self.minimum_agreement:
                continue

            consensus_results.append(
                ConsensusResult(
                    document=document,
                    agreement=agreement,
                    votes=votes[document_id],
                    metadata={
                        "source_count": len(self.sources),
                        "weighted_support": weighted_support[document_id],
                        "total_weight": self._total_weight,
                    },
                )
            )

        consensus_results.sort(
            key=self._sort_key,
            reverse=True,
        )

        return [
            RetrievalResult(
                document=result.document,
                score=result.agreement,
                retriever=self.name,
                metadata={
                    "agreement": result.agreement,
                    "votes": [
                        vote.to_dict()
                        for vote in result.votes
                    ],
                    **result.metadata,
                },
            )
            for result in consensus_results[:limit]
        ]

    @staticmethod
    def _sort_key(
        result: ConsensusResult,
    ) -> tuple[float, float]:
        if not result.votes:
            return result.agreement, 0.0

        average_rank = sum(
            vote.rank for vote in result.votes
        ) / len(result.votes)

        rank_quality = 1.0 / average_rank

        return result.agreement, rank_quality