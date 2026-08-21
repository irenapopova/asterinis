from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .documents import RetrievalResult


ScoreFunction = Callable[[RetrievalResult], float]


@dataclass(slots=True)
class SourceScore:
    document_id: str
    retrieval_score: float
    selection_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "retrieval_score": self.retrieval_score,
            "selection_score": self.selection_score,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SourceSelectionResult:
    selected: list[RetrievalResult]
    rejected: list[RetrievalResult]
    scores: list[SourceScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "selected": [
                result.to_dict()
                for result in self.selected
            ],
            "rejected": [
                result.to_dict()
                for result in self.rejected
            ],
            "scores": [
                score.to_dict()
                for score in self.scores
            ],
            "metadata": dict(self.metadata),
        }


class SourceSelector:
    """
    Selects the strongest retrieval results for downstream use.

    A custom scoring function can combine retrieval score, verification
    information, entity consistency, source quality, or other metadata.
    """

    def __init__(
        self,
        *,
        score_function: ScoreFunction | None = None,
        minimum_score: float = 0.0,
    ) -> None:
        if score_function is not None and not callable(score_function):
            raise TypeError("score_function must be callable.")

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1."
            )

        self.score_function = (
            score_function
            if score_function is not None
            else self._default_score
        )
        self.minimum_score = minimum_score

    def select(
        self,
        results: list[RetrievalResult],
        *,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> SourceSelectionResult:
        if not isinstance(results, list):
            raise TypeError(
                "results must be a list of RetrievalResult objects."
            )

        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        ranked: list[
            tuple[RetrievalResult, float]
        ] = []

        scores: list[SourceScore] = []

        for result in results:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "Every result must be a RetrievalResult."
                )

            selection_score = float(
                self.score_function(result)
            )

            if not 0.0 <= selection_score <= 1.0:
                raise ValueError(
                    "score_function must return a value between 0 and 1."
                )

            scores.append(
                SourceScore(
                    document_id=result.document.id,
                    retrieval_score=float(result.score),
                    selection_score=selection_score,
                    metadata={
                        "retriever": result.retriever,
                    },
                )
            )

            if selection_score >= self.minimum_score:
                ranked.append(
                    (
                        result,
                        selection_score,
                    )
                )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        selected = [
            result
            for result, _ in ranked[:limit]
        ]

        selected_ids = {
            result.document.id
            for result in selected
        }

        rejected = [
            result
            for result in results
            if result.document.id not in selected_ids
        ]

        return SourceSelectionResult(
            selected=selected,
            rejected=rejected,
            scores=scores,
            metadata={
                **(metadata or {}),
                "limit": limit,
                "minimum_score": self.minimum_score,
            },
        )

    @staticmethod
    def _default_score(
        result: RetrievalResult,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(result.score),
            ),
        )