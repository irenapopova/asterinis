from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from asterinis.exceptions import RetrievalError


RewriteStrategy = Callable[[str, dict[str, Any]], str]


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    changed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "changed": self.changed,
            "metadata": dict(self.metadata),
        }


class QueryRewriter:
    """
    Rewrites a query before retrieval.

    The rewriting strategy is supplied by the application. It may use
    deterministic rules, an NLP model, a local model, or an LLM.
    """

    def __init__(
        self,
        strategy: RewriteStrategy,
    ) -> None:
        if not callable(strategy):
            raise TypeError("strategy must be callable.")

        self.strategy = strategy

    def rewrite(
        self,
        query: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryRewriteResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        original = query.strip()

        if not original:
            raise ValueError("query cannot be empty.")

        context = metadata or {}

        try:
            rewritten = self.strategy(
                original,
                context,
            )
        except Exception as exc:
            raise RetrievalError(
                "Query rewriting failed."
            ) from exc

        if not isinstance(rewritten, str):
            raise TypeError(
                "Query rewrite strategy must return a string."
            )

        rewritten = rewritten.strip()

        if not rewritten:
            raise RetrievalError(
                "Query rewrite strategy returned an empty query."
            )

        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            changed=rewritten != original,
            metadata=context,
        )