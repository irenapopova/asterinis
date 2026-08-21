from collections.abc import Callable

from .base import Retriever
from .documents import RetrievalResult


EntityExtractor = Callable[[str], list[str]]


class EntityAwareRetriever(Retriever):
    """
    Enhances retrieval using entities extracted from the query.

    The actual entity extraction implementation is injected,
    keeping the retriever independent from any NLP library.
    """

    name = "entity-aware"

    def __init__(
        self,
        retriever: Retriever,
        entity_extractor: EntityExtractor,
        *,
        entity_boost: float = 0.15,
    ) -> None:

        self._retriever = retriever
        self._entity_extractor = entity_extractor
        self.entity_boost = entity_boost

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        query = self.validate_query(query, limit)

        entities = self._entity_extractor(query)

        results = self._retriever.retrieve(
            query,
            limit=limit * 2,
        )

        normalized_entities = [
            entity.lower()
            for entity in entities
        ]

        enhanced: list[RetrievalResult] = []

        for result in results:

            document_text = result.document.text.lower()

            matched_entities = [
                entity
                for entity in normalized_entities
                if entity in document_text
            ]

            boost = (
                len(matched_entities)
                * self.entity_boost
            )

            enhanced.append(
                RetrievalResult(
                    document=result.document,
                    score=result.score + boost,
                    retriever=self.name,
                    metadata={
                        **result.metadata,
                        "entities": entities,
                        "matched_entities": matched_entities,
                        "base_retriever": result.retriever,
                    },
                )
            )

        return sorted(
            enhanced,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]