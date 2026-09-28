from __future__ import annotations

from typing import Any

from ..base import NLPProvider
from ..classification import TextClassifier
from ..result import Classification, NLPResult


class TextClassifierProvider(NLPProvider):
    """Adapt ``TextClassifier`` to the common NLP provider interface."""

    def __init__(
        self,
        classifier: TextClassifier,
        *,
        name: str = "text-classifier",
    ) -> None:
        if not isinstance(classifier, TextClassifier):
            raise TypeError("classifier must be a TextClassifier.")

        self.classifier = classifier
        self.name = name.strip()

        if not self.name:
            raise ValueError("name cannot be empty.")

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

        predictions = self.classifier.classify(text)
        classifications = [
            Classification(
                label=prediction.label,
                confidence=prediction.confidence,
                metadata=dict(prediction.metadata),
            )
            for prediction in predictions
        ]
        confidences = [
            item.confidence
            for item in classifications
            if item.confidence is not None
        ]

        return NLPResult(
            text=text,
            classifications=classifications,
            confidence=(
                sum(confidences) / len(confidences)
                if confidences
                else None
            ),
            metadata={
                "provider": self.name,
                **kwargs,
            },
        )
