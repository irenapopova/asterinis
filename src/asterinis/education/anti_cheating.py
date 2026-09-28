"""Anti-cheating reports built on static code similarity."""

from __future__ import annotations

from dataclasses import dataclass

from .code_analysis import CodeSimilarityAnalyzer, SimilarSubmission


@dataclass(frozen=True, slots=True)
class AntiCheatingReport:
    exercise_id: str
    matches: tuple[SimilarSubmission, ...]
    reviewed: bool = False


class AntiCheatingAnalyzer:
    def __init__(self, analyzer: CodeSimilarityAnalyzer | None = None) -> None:
        self.analyzer = analyzer or CodeSimilarityAnalyzer()

    def report(self, exercise_id: str, submissions: dict[str, str], *, threshold: float = 0.85) -> AntiCheatingReport:
        return AntiCheatingReport(exercise_id, self.analyzer.find_similar(submissions, threshold=threshold))
