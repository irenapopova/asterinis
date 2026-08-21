from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from asterinis.rag.documents import RetrievalResult


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class RetrievalMetrics:
    """
    Summary metrics for a set of retrieval results.
    """

    count: int
    top_score: float
    average_score: float
    minimum_score: float
    maximum_score: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "top_score": self.top_score,
            "average_score": self.average_score,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
        }


def retrieval_metrics(
    results: Iterable[RetrievalResult],
) -> RetrievalMetrics:
    """
    Calculate basic score statistics for retrieval results.
    """

    items = list(results)

    if not items:
        return RetrievalMetrics(
            count=0,
            top_score=0.0,
            average_score=0.0,
            minimum_score=0.0,
            maximum_score=0.0,
        )

    scores = [_clamp(result.score) for result in items]

    return RetrievalMetrics(
        count=len(scores),
        top_score=max(scores),
        average_score=mean(scores),
        minimum_score=min(scores),
        maximum_score=max(scores),
    )


def precision_at_k(
    relevant: Iterable[bool],
    *,
    k: int,
) -> float:
    """
    Calculate precision@k from relevance judgments.

    Each boolean represents whether a retrieved result
    is considered relevant.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    judgments = list(relevant)[:k]

    if not judgments:
        return 0.0

    relevant_count = sum(judgments)

    return relevant_count / len(judgments)


def reciprocal_rank(
    relevant: Iterable[bool],
) -> float:
    """
    Return the reciprocal rank of the first relevant result.
    """

    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / rank

    return 0.0


def entity_overlap(
    query_entities: Iterable[str],
    document_entities: Iterable[str],
) -> float:
    """
    Measure normalized overlap between query and document entities.
    """

    query_set = {
        entity.strip().lower()
        for entity in query_entities
        if entity.strip()
    }

    document_set = {
        entity.strip().lower()
        for entity in document_entities
        if entity.strip()
    }

    if not query_set:
        return 0.0

    overlap = query_set & document_set

    return len(overlap) / len(query_set)