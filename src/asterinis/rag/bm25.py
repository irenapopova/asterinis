from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from .base import Retriever
from .documents import Document, RetrievalResult


_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
    ]


class BM25Retriever(Retriever):
    """
    Lightweight BM25 retriever for in-memory document collections.

    The implementation has no external dependencies and is intended for
    local retrieval, experiments, tests, and small to medium collections.
    """

    name = "bm25"

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")

        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)
        self.documents = list(documents)

        self._document_tokens = [
            _tokenize(document.text)
            for document in self.documents
        ]

        self._document_frequencies = self._build_document_frequencies()

        self._average_document_length = (
            sum(len(tokens) for tokens in self._document_tokens)
            / len(self._document_tokens)
            if self._document_tokens
            else 0.0
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query = self.validate_query(query, limit)

        if not self.documents:
            return []

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        ranked: list[RetrievalResult] = []

        for document, tokens in zip(
            self.documents,
            self._document_tokens,
            strict=True,
        ):
            score = self._score_document(
                query_tokens,
                tokens,
            )

            if score <= 0:
                continue

            ranked.append(
                RetrievalResult(
                    document=document,
                    score=score,
                    retriever=self.name,
                    metadata={
                        "algorithm": "BM25",
                        "k1": self.k1,
                        "b": self.b,
                    },
                )
            )

        ranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked[:limit]

    def _score_document(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
    ) -> float:
        if not document_tokens:
            return 0.0

        term_frequencies = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0

        for term in set(query_tokens):
            frequency = term_frequencies.get(term, 0)

            if frequency == 0:
                continue

            inverse_document_frequency = self._idf(term)

            length_normalization = (
                1.0 - self.b
                + self.b
                * document_length
                / max(self._average_document_length, 1.0)
            )

            numerator = frequency * (self.k1 + 1.0)
            denominator = (
                frequency
                + self.k1 * length_normalization
            )

            score += (
                inverse_document_frequency
                * numerator
                / denominator
            )

        return score

    def _idf(self, term: str) -> float:
        document_count = len(self.documents)
        frequency = self._document_frequencies.get(term, 0)

        return math.log(
            1.0
            + (
                document_count - frequency + 0.5
            )
            / (
                frequency + 0.5
            )
        )

    def _build_document_frequencies(
        self,
    ) -> dict[str, int]:
        frequencies: Counter[str] = Counter()

        for tokens in self._document_tokens:
            frequencies.update(set(tokens))

        return dict(frequencies)