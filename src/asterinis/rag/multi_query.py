from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import Retriever
from .documents import Document, RetrievalResult


QueryGenerator = Callable[[str], list[str]]


@dataclass(slots=True)
class QueryMatch:
    query: str
    rank: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rank": self.rank,
            "score": self.score,
        }


@dataclass(slots=True)
class MultiQueryCandidate:
    document: Document
    score: float
    matches: list[QueryMatch] = field(default_factory=list)


class MultiQueryRetriever(Retriever):
    """
    Runs several retrieval variants for the same request and merges
    their results.

    This is useful when one formulation of a query may miss documents
    that another formulation retrieves successfully.
    """

    name = "multi-query"

    def __init__(
        self,
        retriever: Retriever,
        query_generator: QueryGenerator,
        *,
        max_queries: int = 5,
        candidate_limit: int = 20,
        rank_constant: int = 60,
        include_original: bool = True,
    ) -> None:
        if not callable(query_generator):
            raise TypeError("query_generator must be callable.")

        if max_queries < 1:
            raise ValueError("max_queries must be greater than zero.")

        if candidate_limit < 1:
            raise ValueError(
                "candidate_limit must be greater than zero."
            )

        if rank_constant < 1:
            raise ValueError(
                "rank_constant must be greater than zero."
            )

        self.retriever = retriever
        self.query_generator = query_generator
        self.max_queries = max_queries
        self.candidate_limit = candidate_limit
        self.rank_constant = rank_constant
        self.include_original = include_original

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query = self.validate_query(query, limit)

        queries = self._build_queries(query)

        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        matches: dict[str, list[QueryMatch]] = defaultdict(list)

        for variant in queries:
            results = self.retriever.retrieve(
                variant,
                limit=self.candidate_limit,
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                document_id = result.document.id

                scores[document_id] += (
                    1.0 / (self.rank_constant + rank)
                )

                documents[document_id] = result.document

                matches[document_id].append(
                    QueryMatch(
                        query=variant,
                        rank=rank,
                        score=float(result.score),
                    )
                )

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            RetrievalResult(
                document=documents[document_id],
                score=scores[document_id],
                retriever=self.name,
                metadata={
                    "query_matches": [
                        match.to_dict()
                        for match in matches[document_id]
                    ],
                    "query_count": len(queries),
                },
            )
            for document_id in ranked_ids[:limit]
        ]

    def _build_queries(
        self,
        original_query: str,
    ) -> list[str]:
        generated = self.query_generator(
            original_query
        )

        if not isinstance(generated, list):
            raise TypeError(
                "query_generator must return a list of strings."
            )

        queries: list[str] = []

        if self.include_original:
            queries.append(original_query)

        for query in generated:
            if not isinstance(query, str):
                raise TypeError(
                    "Each generated query must be a string."
                )

            query = query.strip()

            if not query:
                continue

            if query not in queries:
                queries.append(query)

            if len(queries) >= self.max_queries:
                break

        return queries