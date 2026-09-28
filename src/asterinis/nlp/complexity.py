from __future__ import annotations

import re
from dataclasses import dataclass


_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(slots=True)
class ComplexityResult:
    score: float
    word_count: int
    sentence_count: int
    conjunction_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "score": self.score,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "conjunction_count": self.conjunction_count,
        }


class QueryComplexityAnalyzer:
    """
    Lightweight query complexity estimator.

    This is intentionally transparent and deterministic. Applications can
    replace it with a learned model later.
    """

    _CONJUNCTIONS = {
        "and",
        "or",
        "but",
        "while",
        "although",
        "because",
        "compare",
        "versus",
        "vs",
    }

    def analyze(
        self,
        text: str,
    ) -> ComplexityResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        words = [
            word.lower()
            for word in _WORD_PATTERN.findall(text)
        ]

        sentence_count = max(
            1,
            sum(
                text.count(symbol)
                for symbol in ".!?"
            ),
        )

        conjunction_count = sum(
            1
            for word in words
            if word in self._CONJUNCTIONS
        )

        word_component = min(
            len(words) / 40.0,
            1.0,
        )

        sentence_component = min(
            sentence_count / 4.0,
            1.0,
        )

        conjunction_component = min(
            conjunction_count / 4.0,
            1.0,
        )

        score = (
            0.5 * word_component
            + 0.2 * sentence_component
            + 0.3 * conjunction_component
        )

        return ComplexityResult(
            score=min(1.0, score),
            word_count=len(words),
            sentence_count=sentence_count,
            conjunction_count=conjunction_count,
        )