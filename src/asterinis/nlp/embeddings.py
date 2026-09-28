from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from .result import NLPResult


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    model: str


class HashEmbeddingProvider:
    """Deterministic local embeddings with no external ML dependency.

    This is useful for tests, local routing, and prototyping. It is not a
    replacement for a semantic transformer embedding model.
    """

    name = "hash-embeddings"

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be greater than zero.")
        self.dimensions = dimensions

    def embed(self, text: str) -> EmbeddingResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")

        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]

        return EmbeddingResult(text, vector, self.name)

    def analyze(self, text: str, **kwargs: Any) -> NLPResult:
        result = self.embed(text)
        return NLPResult(
            text=result.text,
            metadata={
                "provider": self.name,
                "embedding": result.vector,
                "dimensions": self.dimensions,
                **kwargs,
            },
        )
