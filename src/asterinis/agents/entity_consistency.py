from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class EntityConsistencyResult:
    query_entities: list[str]
    document_entities: list[str]
    matched_entities: list[str]
    missing_entities: list[str]
    unexpected_entities: list[str]
    score: float
    consistent: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_entities": list(self.query_entities),
            "document_entities": list(self.document_entities),
            "matched_entities": list(self.matched_entities),
            "missing_entities": list(self.missing_entities),
            "unexpected_entities": list(self.unexpected_entities),
            "score": self.score,
            "consistent": self.consistent,
            "metadata": dict(self.metadata),
        }


class EntityConsistencyAgent:
    """
    Compares entities from a query with entities found in retrieved evidence.

    The agent is intentionally independent from a specific NLP provider.
    Entity extraction can be handled by Flair, another NLP library,
    or application-specific code before calling this agent.
    """

    name = "entity-consistency"

    def __init__(
        self,
        *,
        consistency_threshold: float = 0.75,
        case_sensitive: bool = False,
    ) -> None:
        if not 0.0 <= consistency_threshold <= 1.0:
            raise ValueError(
                "consistency_threshold must be between 0 and 1."
            )

        self.consistency_threshold = consistency_threshold
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        query_entities: Iterable[str],
        document_entities: Iterable[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EntityConsistencyResult:
        query = self._normalize(query_entities)
        documents = self._normalize(document_entities)

        if not query:
            return EntityConsistencyResult(
                query_entities=[],
                document_entities=documents,
                matched_entities=[],
                missing_entities=[],
                unexpected_entities=documents,
                score=1.0,
                consistent=True,
                metadata=metadata or {},
            )

        query_set = set(query)
        document_set = set(documents)

        matched = sorted(query_set & document_set)
        missing = sorted(query_set - document_set)
        unexpected = sorted(document_set - query_set)

        score = len(matched) / len(query_set)
        consistent = score >= self.consistency_threshold

        return EntityConsistencyResult(
            query_entities=query,
            document_entities=documents,
            matched_entities=matched,
            missing_entities=missing,
            unexpected_entities=unexpected,
            score=score,
            consistent=consistent,
            metadata=metadata or {},
        )

    def _normalize(
        self,
        entities: Iterable[str],
    ) -> list[str]:
        normalized: list[str] = []

        for entity in entities:
            if not isinstance(entity, str):
                raise TypeError("Each entity must be a string.")

            value = entity.strip()

            if not value:
                continue

            if not self.case_sensitive:
                value = value.lower()

            if value not in normalized:
                normalized.append(value)

        return normalized