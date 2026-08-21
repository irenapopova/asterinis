from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from asterinis.rag.documents import RetrievalResult


@dataclass(slots=True)
class RetrievalQualityDecision:
    action: str
    score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalQualityAgent:
    """
    Evaluates whether retrieved evidence is strong enough
    for downstream generation.
    """

    name = "retrieval-quality"

    def __init__(
        self,
        *,
        accept_threshold: float = 0.70,
        retry_threshold: float = 0.40,
    ) -> None:
        if not 0 <= retry_threshold <= accept_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= retry_threshold <= accept_threshold <= 1."
            )

        self.accept_threshold = accept_threshold
        self.retry_threshold = retry_threshold

    def evaluate(
        self,
        results: list[RetrievalResult],
    ) -> RetrievalQualityDecision:
        if not results:
            return RetrievalQualityDecision(
                action="clarify",
                score=0.0,
                reason="No retrieval results were returned.",
            )

        scores = [
            max(0.0, min(1.0, result.score))
            for result in results
        ]

        top_score = max(scores)
        average_score = mean(scores)

        combined_score = (
            0.7 * top_score
            + 0.3 * average_score
        )

        metadata = {
            "top_score": top_score,
            "average_score": average_score,
            "result_count": len(results),
        }

        if combined_score >= self.accept_threshold:
            return RetrievalQualityDecision(
                action="generate",
                score=combined_score,
                reason=(
                    "Retrieved evidence is strong enough "
                    "for downstream generation."
                ),
                metadata=metadata,
            )

        if combined_score >= self.retry_threshold:
            return RetrievalQualityDecision(
                action="retrieve_again",
                score=combined_score,
                reason=(
                    "Retrieved evidence is usable but "
                    "not strong enough yet."
                ),
                metadata=metadata,
            )

        return RetrievalQualityDecision(
            action="clarify",
            score=combined_score,
            reason=(
                "Retrieved evidence is too weak to "
                "support a reliable answer."
            ),
            metadata=metadata,
        )