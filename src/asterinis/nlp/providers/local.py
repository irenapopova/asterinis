from __future__ import annotations

from typing import Any

from ..base import NLPProvider
from ..embeddings import HashEmbeddingProvider
from ..language import LanguageDetector
from ..result import NLPResult
from ..sentiment import SentimentAnalyzer


class LanguageNLPProvider(NLPProvider):
    name = "local-language-detector"

    def __init__(self, detector: LanguageDetector) -> None:
        self.detector = detector

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        result = self.detector.detect(text)
        return NLPResult(
            text=text,
            language=result.language,
            confidence=result.confidence,
            metadata={"provider": self.name, **kwargs},
        )


class SentimentNLPProvider(NLPProvider):
    name = "local-sentiment"

    def __init__(self, analyzer: SentimentAnalyzer | None = None) -> None:
        self.analyzer = analyzer or SentimentAnalyzer()

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        result = self.analyzer.analyze(text)
        return NLPResult(
            text=text,
            intent=result.label,
            confidence=result.confidence,
            metadata={"provider": self.name, "sentiment": result.label, **kwargs},
        )


class EmbeddingNLPProvider(NLPProvider):
    name = "hash-embeddings"

    def __init__(self, provider: HashEmbeddingProvider | None = None) -> None:
        self.provider = provider or HashEmbeddingProvider()

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        return self.provider.analyze(text, **kwargs)
