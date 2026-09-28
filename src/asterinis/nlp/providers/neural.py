"""NLP provider backed by the native Asterinis sequence tagger."""

from __future__ import annotations

from typing import Any

from ..base import NLPProvider
from ..models import ModelCard
from ..native_sequence import NativeSequenceTagger
from ..result import Entity, NLPResult


class NeuralNERProvider(NLPProvider):
    """Expose a native Asterinis NER model through the common provider API."""

    name = "asterinis-neural-ner"

    def __init__(self, model: NativeSequenceTagger, *, model_name: str = "native-ner") -> None:
        if not isinstance(model, NativeSequenceTagger):
            raise TypeError("model must be a NativeSequenceTagger.")
        self.model = model
        self.model_name = model_name.strip()
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")

    @classmethod
    def load(cls, path: str, *, model_name: str = "native-ner") -> "NeuralNERProvider":
        return cls(NativeSequenceTagger.load(path), model_name=model_name)

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            name=self.model_name,
            task="ner",
            description="Asterinis native BiLSTM sequence tagger.",
            source="asterinis.neural",
        )

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        text = text.strip()
        predictions = self.model.predict(text)
        entities: list[Entity] = []
        current: list[Any] = []

        def flush() -> None:
            if current:
                entities.append(
                    Entity(
                        text=text[current[0].start : current[-1].end],
                        label=current[0].label[2:],
                        confidence=sum(item.confidence for item in current) / len(current),
                        start=current[0].start,
                        end=current[-1].end,
                    )
                )
                current.clear()

        for prediction in predictions:
            if prediction.label == "O":
                flush()
            elif prediction.label.startswith("B-"):
                flush()
                current.append(prediction)
            elif prediction.label.startswith("I-"):
                if not current or current[0].label[2:] != prediction.label[2:]:
                    flush()
                current.append(prediction)
            else:
                flush()
        flush()
        confidence = sum(item.confidence for item in entities) / len(entities) if entities else None
        return NLPResult(
            text=text.strip(),
            entities=entities,
            confidence=confidence,
            metadata={"provider": self.name, "model": self.model_name, **kwargs},
        )


__all__ = ["NeuralNERProvider"]
