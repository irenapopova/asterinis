from collections.abc import Callable

from .base import Retriever
from .documents import RetrievalResult


RetrieveFunction = Callable[
    [str, int],
    list[RetrievalResult],
]


class CustomRetriever(Retriever):
    """
    Adapter for user-defined retrieval functions.
    """

    name = "custom"

    def __init__(
        self,
        retrieve_fn: RetrieveFunction,
        *,
        name: str = "custom",
    ) -> None:

        if not callable(retrieve_fn):
            raise TypeError(
                "retrieve_fn must be callable."
            )

        self._retrieve_fn = retrieve_fn
        self.name = name

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        query = self.validate_query(query, limit)

        return self._retrieve_fn(
            query,
            limit,
        )