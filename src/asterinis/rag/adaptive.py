from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import Retriever
from .documents import RetrievalResult


RetrievalDecisionStrategy = Callable[
    [str, dict[str, Any]],
    bool,
]


@dataclass(slots=True)
class AdaptiveRetrievalDecision:
    should_retrieve: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrieve": self.should_retrieve,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class AdaptiveRetriever:
    """
    Adds a decision layer before retrieval.

    The decision strategy can use query text, NLP signals, confidence,
    application metadata, or other runtime information to decide whether
    retrieval is necessary.
    """

    name = "adaptive"

    def __init__(
        self,
        retriever: Retriever,
        decision_strategy: RetrievalDecisionStrategy,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise TypeError(
                "retriever must implement the Asterinis Retriever interface."
            )

        if not callable(decision_strategy):
            raise TypeError(
                "decision_strategy must be callable."
            )

        self.retriever = retriever
        self.decision_strategy = decision_strategy

    def decide(
        self,
        query: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AdaptiveRetrievalDecision:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        context = metadata or {}

        should_retrieve = bool(
            self.decision_strategy(
                query,
                context,
            )
        )

        reason = (
            "Retrieval is required for this request."
            if should_retrieve
            else "Retrieval was skipped for this request."
        )

        return AdaptiveRetrievalDecision(
            should_retrieve=should_retrieve,
            reason=reason,
            metadata=dict(context),
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError(
                "limit must be greater than zero."
            )

        decision = self.decide(
            query,
            metadata=metadata,
        )

        if not decision.should_retrieve:
            return []

        return self.retriever.retrieve(
            query,
            limit=limit,
        )