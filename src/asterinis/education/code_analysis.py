"""Safe static code similarity analysis; never executes submitted code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeSimilarity:
    score: float
    same_structure: bool
    valid_first: bool
    valid_second: bool


@dataclass(frozen=True, slots=True)
class SimilarSubmission:
    first_id: str
    second_id: str
    score: float


class CodeSimilarityAnalyzer:
    """Compare source structure and normalized tokens without running it."""

    def compare(self, first: str, second: str, *, threshold: float = 0.85) -> CodeSimilarity:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        first_tree = self._parse(first)
        second_tree = self._parse(second)
        valid_first = first_tree is not None
        valid_second = second_tree is not None
        if not valid_first or not valid_second:
            return CodeSimilarity(0.0, False, valid_first, valid_second)
        first_tokens = self._normalized_tokens(first_tree)
        second_tokens = self._normalized_tokens(second_tree)
        union = first_tokens | second_tokens
        score = len(first_tokens & second_tokens) / len(union) if union else 1.0
        return CodeSimilarity(score, score >= threshold, True, True)

    def find_similar(
        self,
        submissions: dict[str, str],
        *,
        threshold: float = 0.85,
    ) -> tuple[SimilarSubmission, ...]:
        ids = tuple(submissions)
        matches: list[SimilarSubmission] = []
        for index, first_id in enumerate(ids):
            for second_id in ids[index + 1 :]:
                result = self.compare(submissions[first_id], submissions[second_id], threshold=threshold)
                if result.same_structure:
                    matches.append(SimilarSubmission(first_id, second_id, result.score))
        return tuple(matches)

    @staticmethod
    def _parse(source: str) -> ast.AST | None:
        try:
            return ast.parse(source)
        except (SyntaxError, TypeError):
            return None

    @staticmethod
    def _normalized_tokens(tree: ast.AST) -> set[str]:
        tokens: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                tokens.add("Name")
            elif isinstance(node, ast.Constant):
                tokens.add(f"Constant:{type(node.value).__name__}")
            else:
                tokens.add(type(node).__name__)
        return tokens
