from .base import Retriever
from .context_builder import ContextBuilder
from .documents import RetrievalResult
from .reranker import Reranker


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: Reranker | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = (
            context_builder or ContextBuilder()
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        results = self.retriever.retrieve(
            query,
            limit=limit,
        )

        if self.reranker is not None:
            results = self.reranker.rerank(
                query,
                results,
            )

        return results[:limit]

    def build_context(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> str:

        results = self.retrieve(
            query,
            limit=limit,
        )

        return self.context_builder.build(
            results
        )