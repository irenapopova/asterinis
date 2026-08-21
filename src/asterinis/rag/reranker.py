from abc import ABC, abstractmethod

from .documents import RetrievalResult


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class ScoreReranker(Reranker):
    name = "score"

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )