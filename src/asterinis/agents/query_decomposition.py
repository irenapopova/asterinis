from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class QueryDecompositionResult:
    query: str
    subqueries: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.subqueries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "subqueries": list(self.subqueries),
            "count": self.count,
            "metadata": dict(self.metadata),
        }


class QueryDecompositionAgent:
    """
    Splits a complex query into smaller retrieval-friendly subqueries.

    The decomposition strategy is injected so applications can use a
    rule-based function, an NLP model, or an LLM without coupling the agent
    to one provider.
    """

    name = "query-decomposition"

    def __init__(
        self,
        decomposer: Callable[[str], list[str]],
        *,
        max_subqueries: int = 8,
    ) -> None:
        if not callable(decomposer):
            raise TypeError("decomposer must be callable.")

        if max_subqueries < 1:
            raise ValueError("max_subqueries must be greater than zero.")

        self.decomposer = decomposer
        self.max_subqueries = max_subqueries

    def run(
        self,
        query: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryDecompositionResult:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        raw_subqueries = self.decomposer(query)

        if not isinstance(raw_subqueries, list):
            raise TypeError("decomposer must return a list of strings.")

        subqueries = self._normalize_subqueries(
            raw_subqueries,
            original_query=query,
        )

        return QueryDecompositionResult(
            query=query,
            subqueries=subqueries,
            metadata=metadata or {},
        )

    def _normalize_subqueries(
        self,
        subqueries: list[str],
        *,
        original_query: str,
    ) -> list[str]:
        normalized: list[str] = []

        for item in subqueries:
            if not isinstance(item, str):
                raise TypeError("Each subquery must be a string.")

            item = item.strip()

            if not item:
                continue

            if item not in normalized:
                normalized.append(item)

            if len(normalized) >= self.max_subqueries:
                break

        if not normalized:
            return [original_query]

        return normalized