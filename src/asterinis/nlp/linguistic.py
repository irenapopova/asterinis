from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"[.!?]+")


@dataclass(slots=True)
class LinguisticFeatures:
    word_count: int
    unique_word_count: int
    sentence_count: int
    average_word_length: float
    lexical_diversity: float
    question_count: int
    punctuation_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "unique_word_count": self.unique_word_count,
            "sentence_count": self.sentence_count,
            "average_word_length": self.average_word_length,
            "lexical_diversity": self.lexical_diversity,
            "question_count": self.question_count,
            "punctuation_count": self.punctuation_count,
            "metadata": dict(self.metadata),
        }


class LinguisticAnalyzer:
    """
    Extracts lightweight, deterministic linguistic features from text.

    These features are not intended to replace full parsing or linguistic
    models. They provide inexpensive signals that can be used by routing,
    retrieval, complexity estimation, evaluation, or adaptive strategies.
    """

    def analyze(
        self,
        text: str,
    ) -> LinguisticFeatures:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        words = _WORD_PATTERN.findall(text)

        normalized_words = [
            word.lower()
            for word in words
        ]

        word_count = len(words)
        unique_word_count = len(
            set(normalized_words)
        )

        sentence_markers = _SENTENCE_PATTERN.findall(
            text
        )

        sentence_count = max(
            1,
            len(sentence_markers),
        )

        average_word_length = (
            sum(len(word) for word in words)
            / word_count
            if word_count
            else 0.0
        )

        lexical_diversity = (
            unique_word_count / word_count
            if word_count
            else 0.0
        )

        question_count = text.count("?")

        punctuation_count = sum(
            text.count(symbol)
            for symbol in ".,!?;:"
        )

        return LinguisticFeatures(
            word_count=word_count,
            unique_word_count=unique_word_count,
            sentence_count=sentence_count,
            average_word_length=average_word_length,
            lexical_diversity=lexical_diversity,
            question_count=question_count,
            punctuation_count=punctuation_count,
        )