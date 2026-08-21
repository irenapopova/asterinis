from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from asterinis.exceptions import RetrievalError

from .base import Retriever
from .documents import RetrievalResult


QualityEvaluator = Callable[[list[RetrievalResult]], float]
QueryRefiner = Callable[[str, int, list[RetrievalResult]], str]


@dataclass(slots=True)
class RetrievalAttempt:
    attempt: int
    query: str
    quality_score: float
    result_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "query": self.query,
            "quality_score": self.quality_score,
            "result_count": self.result_count,
        }


@dataclass(slots=True)
class IterativeRetrievalResult:
    original_query: str
    final_query: str
    results: list[RetrievalResult]
    attempts: list[RetrievalAttempt]
    quality_score: float
    satisfied: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "final_query": self.final_query,
            "quality_score": self.quality_score,
            "satisfied": self.satisfied,
            "attempt_count": self.attempt_count,
            "attempts": [
                attempt.to_dict()
                for attempt in self.attempts
            ],
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "metadata": dict(self.metadata),
        }


class IterativeRetriever:
    """
    Repeats retrieval when the returned evidence does not meet a configured
    quality threshold.

    The retriever itself does not decide how a query should be improved.
    Instead, the caller supplies:

    - a quality evaluator that scores retrieval results between 0 and 1
    - a query refiner that produces the next query when another attempt
      is needed

    This keeps the class independent from a specific NLP model or LLM.
    """

    name = "iterative"

    def __init__(
        self,
        retriever: Retriever,
        quality_evaluator: QualityEvaluator,
        query_refiner: QueryRefiner,
        *,
        quality_threshold: float = 0.70,
        max_attempts: int = 3,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise TypeError(
                "retriever must implement the Asterinis Retriever interface."
            )

        if not callable(quality_evaluator):
            raise TypeError(
                "quality_evaluator must be callable."
            )

        if not callable(query_refiner):
            raise TypeError(
                "query_refiner must be callable."
            )

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError(
                "quality_threshold must be between 0 and 1."
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be greater than zero."
            )

        self.retriever = retriever
        self.quality_evaluator = quality_evaluator
        self.query_refiner = query_refiner
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> IterativeRetrievalResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        if limit < 1:
            raise ValueError(
                "limit must be greater than zero."
            )

        original_query = query
        current_query = query
        attempts: list[RetrievalAttempt] = []

        last_results: list[RetrievalResult] = []
        last_quality = 0.0

        for attempt_number in range(1, self.max_attempts + 1):
            results = self.retriever.retrieve(
                current_query,
                limit=limit,
            )

            quality = float(
                self.quality_evaluator(results)
            )

            if not 0.0 <= quality <= 1.0:
                raise RetrievalError(
                    "quality_evaluator must return a value between 0 and 1."
                )

            attempts.append(
                RetrievalAttempt(
                    attempt=attempt_number,
                    query=current_query,
                    quality_score=quality,
                    result_count=len(results),
                )
            )

            last_results = results
            last_quality = quality

            if quality >= self.quality_threshold:
                return IterativeRetrievalResult(
                    original_query=original_query,
                    final_query=current_query,
                    results=results,
                    attempts=attempts,
                    quality_score=quality,
                    satisfied=True,
                    metadata=dict(metadata or {}),
                )

            if attempt_number == self.max_attempts:
                break

            refined_query = self.query_refiner(
                current_query,
                attempt_number,
                results,
            )

            if not isinstance(refined_query, str):
                raise TypeError(
                    "query_refiner must return a string."
                )

            refined_query = refined_query.strip()

            if not refined_query:
                raise RetrievalError(
                    "query_refiner returned an empty query."
                )

            current_query = refined_query

        return IterativeRetrievalResult(
            original_query=original_query,
            final_query=current_query,
            results=last_results,
            attempts=attempts,
            quality_score=last_quality,
            satisfied=False,
            metadata=dict(metadata or {}),
        )