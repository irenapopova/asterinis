from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .documents import RetrievalResult


SourceCheck = Callable[
    [RetrievalResult],
    tuple[bool, float, str],
]


@dataclass(slots=True)
class SourceVerificationResult:
    document_id: str
    accepted: bool
    score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "accepted": self.accepted,
            "score": self.score,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class SourceVerifier:
    """
    Applies an application-defined quality policy to retrieved sources.

    Verification may consider provenance, metadata, domain rules, recency,
    trust scores, or other information available to the application.
    """

    def __init__(
        self,
        check: SourceCheck,
    ) -> None:
        if not callable(check):
            raise TypeError("check must be callable.")

        self.check = check

    def verify(
        self,
        result: RetrievalResult,
    ) -> SourceVerificationResult:
        accepted, score, reason = self.check(
            result
        )

        score = float(score)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Source verification score must be between 0 and 1."
            )

        if not isinstance(reason, str):
            raise TypeError(
                "Source verification reason must be a string."
            )

        return SourceVerificationResult(
            document_id=result.document.id,
            accepted=bool(accepted),
            score=score,
            reason=reason.strip(),
            metadata={
                "retriever": result.retriever,
                "retrieval_score": result.score,
            },
        )