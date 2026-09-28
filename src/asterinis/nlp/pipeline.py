from __future__ import annotations

from typing import Any

from .base import NLPProvider
from .classification import TextClassifier
from .complexity import QueryComplexityAnalyzer
from .intent import IntentDetector
from .language import LanguageDetector
from .linguistic import LinguisticAnalyzer
from .result import Classification, NLPResult


class NLPPipeline:
    """
    Coordinates provider-based NLP with optional Asterinis analyzers.

    Provider output remains the primary NLP result. Additional analyzers can
    enrich the result with intent, language, classification, complexity, and
    lightweight linguistic features.
    """

    def __init__(
        self,
        provider: NLPProvider,
        *,
        classifier: TextClassifier | None = None,
        intent_detector: IntentDetector | None = None,
        language_detector: LanguageDetector | None = None,
        complexity_analyzer: QueryComplexityAnalyzer | None = None,
        linguistic_analyzer: LinguisticAnalyzer | None = None,
    ) -> None:
        if not isinstance(provider, NLPProvider):
            raise TypeError(
                "provider must implement the Asterinis NLPProvider interface."
            )

        self.provider = provider
        self.classifier = classifier
        self.intent_detector = intent_detector
        self.language_detector = language_detector

        self.complexity_analyzer = (
            complexity_analyzer
            or QueryComplexityAnalyzer()
        )

        self.linguistic_analyzer = (
            linguistic_analyzer
            or LinguisticAnalyzer()
        )

    def analyze(
        self,
        text: str,
        **kwargs: Any,
    ) -> NLPResult:
        result = self.provider.analyze(
            text,
            **kwargs,
        )

        if self.classifier is not None:
            classifications = self.classifier.classify(
                text
            )

            result.classifications = [
                Classification(
                    label=item.label,
                    confidence=item.confidence,
                    metadata=item.metadata,
                )
                for item in classifications
            ]

        if self.intent_detector is not None:
            intent = self.intent_detector.detect(
                text
            )

            result.intent = intent.intent

            result.metadata["intent"] = {
                "confidence": intent.confidence,
            }

        if self.language_detector is not None:
            language = self.language_detector.detect(
                text
            )

            result.language = language.language

            result.metadata["language"] = {
                "confidence": language.confidence,
            }

        if result.complexity is None:
            complexity = self.complexity_analyzer.analyze(
                text
            )

            result.complexity = complexity.score

            result.metadata["complexity"] = (
                complexity.to_dict()
            )

        linguistic = self.linguistic_analyzer.analyze(
            text
        )

        result.metadata["linguistic"] = (
            linguistic.to_dict()
        )

        return result