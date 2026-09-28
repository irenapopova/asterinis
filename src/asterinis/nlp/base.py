from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .result import NLPResult


class NLPProvider(ABC):
    """
    Base interface for NLP providers used by Asterinis.

    Providers may wrap Flair, local models, external APIs, or custom
    application-specific NLP implementations.
    """

    name: str = "nlp-provider"

    @abstractmethod
    def analyze(
        self,
        text: str,
        **kwargs: Any,
    ) -> NLPResult:
        """Analyze text and return a normalized Asterinis NLP result."""
        raise NotImplementedError