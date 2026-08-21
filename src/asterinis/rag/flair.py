from .base import Retriever
from .documents import RetrievalResult
from .entity import EntityAwareRetriever


class FlairRetriever(Retriever):
    """
    Retrieval wrapper that uses Flair entity recognition
    to enrich another Asterinis retriever.
    """

    name = "flair"

    def __init__(
        self,
        retriever: Retriever,
        *,
        model_name: str = "ner",
        entity_boost: float = 0.15,
    ) -> None:

        try:
            from flair.nn import Classifier
        except ImportError as exc:
            raise ImportError(
                "Flair support is not installed. "
                'Install it with: pip install "asterinis[flair]"'
            ) from exc

        self.model_name = model_name
        self._model = Classifier.load(model_name)

        self._retriever = EntityAwareRetriever(
            retriever,
            self._extract_entities,
            entity_boost=entity_boost,
        )

    def _extract_entities(
        self,
        text: str,
    ) -> list[str]:

        from flair.data import Sentence

        sentence = Sentence(text)

        self._model.predict(sentence)

        return [
            span.text
            for span in sentence.get_spans("ner")
        ]

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:

        results = self._retriever.retrieve(
            query,
            limit=limit,
        )

        return [
            RetrievalResult(
                document=result.document,
                score=result.score,
                retriever=self.name,
                metadata={
                    **result.metadata,
                    "flair_model": self.model_name,
                },
            )
            for result in results
        ]