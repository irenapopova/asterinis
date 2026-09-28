from __future__ import annotations

from typing import Any

from ..base import NLPProvider
from ..embeddings import EmbeddingResult
from ..result import NLPResult


class SentenceTransformerEmbeddingProvider(NLPProvider):
    """Embedding provider backed by sentence-transformers."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Sentence Transformer support requires the optional "
                    'dependency: pip install "asterinis[embeddings]"'
                ) from exc
            model = SentenceTransformer(model_name)

        self.model_name = model_name
        self.model = model

    def embed(self, text: str) -> EmbeddingResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")

        encoded = self.model.encode(text)
        vector = [float(value) for value in encoded]
        return EmbeddingResult(text, vector, self.model_name)

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        result = self.embed(text)
        return NLPResult(
            text=result.text,
            metadata={
                "provider": self.name,
                "model": self.model_name,
                "embedding": result.vector,
                "dimensions": len(result.vector),
                **kwargs,
            },
        )


class TransformersSentimentProvider(NLPProvider):
    """Sentiment provider backed by a Transformers pipeline."""

    name = "transformers-sentiment"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        *,
        pipeline: Any | None = None,
    ) -> None:
        if pipeline is None:
            try:
                from transformers import pipeline as create_pipeline
            except ImportError as exc:
                raise ImportError(
                    "Transformers support requires the optional dependency: "
                    'pip install "asterinis[transformers]"'
                ) from exc
            pipeline = create_pipeline(
                "sentiment-analysis",
                model=model_name,
            )

        self.model_name = model_name
        self.pipeline = pipeline

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")

        prediction = self.pipeline(text)[0]
        label = str(prediction["label"]).lower()
        confidence = float(prediction["score"])

        return NLPResult(
            text=text,
            intent=label,
            confidence=confidence,
            metadata={
                "provider": self.name,
                "model": self.model_name,
                "sentiment": label,
                **kwargs,
            },
        )


class FastTextLanguageProvider(NLPProvider):
    """Language detection provider backed by a FastText model."""

    name = "fasttext-language"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        if model is None:
            if not model_path:
                raise ValueError("model_path is required when model is not supplied.")
            try:
                import fasttext
            except ImportError as exc:
                raise ImportError(
                    "FastText support requires the optional dependency: "
                    'pip install "asterinis[fasttext]"'
                ) from exc
            model = fasttext.load_model(model_path)

        self.model_path = model_path
        self.model = model

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")

        labels, probabilities = self.model.predict(text, k=1)
        language = str(labels[0]).removeprefix("__label__")
        confidence = float(probabilities[0])

        return NLPResult(
            text=text,
            language=language,
            confidence=confidence,
            metadata={
                "provider": self.name,
                "model_path": self.model_path,
                **kwargs,
            },
        )
