from abc import ABC, abstractmethod

from .documents import RetrievalResult


class Retriever(ABC):
    """
    Base interface implemented by all Asterinis retrievers.
    """

    name: str = "retriever"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    @staticmethod
    def validate_query(query: str, limit: int) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        return query