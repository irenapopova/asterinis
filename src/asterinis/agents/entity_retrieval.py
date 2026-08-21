from dataclasses import dataclass, field
from typing import Any

from asterinis.rag.base import Retriever
from asterinis.rag.documents import RetrievalResult


@dataclass(slots=True)
class EntityRetrievalResult:
    query: str
    entities: list[str]
    results: list[RetrievalResult]
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityRetrievalAgent:
    """
    Uses extracted entities to enrich retrieval decisions.

    Entity extraction is injected so the agent is not tied
    to one NLP framework.
    """

    name = "entity-retrieval"

    def __init__(
        self,
        retriever: Retriever,
        entity_extractor,
    ) -> None:
        if not callable(entity_extractor):
            raise TypeError(
                "entity_extractor must be callable."
            )

        self.retriever = retriever
        self.entity_extractor = entity_extractor

    def run(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> EntityRetrievalResult:
        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        entities = self.entity_extractor(query)

        enriched_query = self._build_enriched_query(
            query,
            entities,
        )

        results = self.retriever.retrieve(
            enriched_query,
            limit=limit,
        )

        return EntityRetrievalResult(
            query=query,
            entities=entities,
            results=results,
            metadata={
                "enriched_query": enriched_query,
            },
        )

    @staticmethod
    def _build_enriched_query(
        query: str,
        entities: list[str],
    ) -> str:
        if not entities:
            return query

        unique_entities = list(dict.fromkeys(entities))

        entity_text = " ".join(unique_entities)

        return f"{query} {entity_text}".strip()