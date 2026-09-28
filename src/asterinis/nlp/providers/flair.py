from __future__ import annotations

from typing import Any

from ..base import NLPProvider
from ..models import ModelCard, normalize_confidence
from ..result import Entity, NLPResult


class FlairNLPProvider(NLPProvider):
    """
    Optional NLP provider backed by Flair.

    Flair remains an external dependency and is loaded only when this
    provider is instantiated.
    """

    name = "flair"

    def __init__(
        self,
        model_name: str = "ner",
    ) -> None:
        try:
            from flair.nn import Classifier
        except ImportError as exc:
            raise ImportError(
                "Flair NLP support requires the optional dependency. "
                'Install it with: pip install "asterinis[flair]"'
            ) from exc

        self.model_name = model_name
        self._classifier = Classifier
        self._model = None

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            name=self.model_name,
            task="ner",
            description="Flair sequence tagging model.",
            source="https://github.com/flairNLP/flair",
            license="MIT",
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is None:
            self._model = self._classifier.load(self.model_name)

    def analyze(
        self,
        text: str,
        **kwargs: Any,
    ) -> NLPResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        self.load()
        from flair.data import Sentence

        sentence = Sentence(text)
        self._model.predict(sentence)

        entities: list[Entity] = []

        for span in sentence.get_spans("ner"):
            label = span.get_label("ner")

            entities.append(
                Entity(
                    text=span.text,
                    label=label.value,
                    confidence=normalize_confidence(float(label.score)),
                    metadata={
                        "provider": self.name,
                        "model": self.model_name,
                    },
                )
            )

        confidence = (
            sum(
                entity.confidence or 0.0
                for entity in entities
            )
            / len(entities)
            if entities
            else None
        )

        return NLPResult(
            text=text,
            entities=entities,
            confidence=confidence,
            metadata={
                "provider": self.name,
                "model": self.model_name,
            },
        )

    def analyze_many(self, texts: list[str], **kwargs: Any) -> list[NLPResult]:
        return [self.analyze(text, **kwargs) for text in texts]
