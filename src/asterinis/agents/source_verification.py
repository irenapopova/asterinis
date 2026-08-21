from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asterinis.rag.documents import RetrievalResult
from asterinis.rag.source_verification import (
    SourceVerificationResult,
    SourceVerifier,
)


@dataclass(slots=True)
class SourceVerificationSummary:
    accepted: list[SourceVerificationResult]
    rejected: list[SourceVerificationResult]
    average_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def total_count(self) -> int:
        return self.accepted_count + self.rejected_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [
                result.to_dict()
                for result in self.accepted
            ],
            "rejected": [
                result.to_dict()
                for result in self.rejected
            ],
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "total_count": self.total_count,
            "average_score": self.average_score,
            "metadata": dict(self.metadata),
        }


class SourceVerificationAgent:
    """
    Evaluates retrieved evidence before it is used downstream.

    The agent separates accepted and rejected sources using a SourceVerifier,
    making source-quality decisions explicit and inspectable.
    """

    name = "source-verification"

    def __init__(
        self,
        verifier: SourceVerifier,
    ) -> None:
        if not isinstance(verifier, SourceVerifier):
            raise TypeError(
                "verifier must be a SourceVerifier."
            )

        self.verifier = verifier

    def evaluate(
        self,
        results: list[RetrievalResult],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SourceVerificationSummary:
        if not isinstance(results, list):
            raise TypeError(
                "results must be a list of RetrievalResult objects."
            )

        verified: list[SourceVerificationResult] = []

        for result in results:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "Every result must be a RetrievalResult."
                )

            verified.append(
                self.verifier.verify(result)
            )

        accepted = [
            result
            for result in verified
            if result.accepted
        ]

        rejected = [
            result
            for result in verified
            if not result.accepted
        ]

        average_score = (
            sum(result.score for result in verified)
            / len(verified)
            if verified
            else 0.0
        )

        return SourceVerificationSummary(
            accepted=accepted,
            rejected=rejected,
            average_score=average_score,
            metadata=dict(metadata or {}),
        )