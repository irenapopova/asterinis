"""Actionable coding feedback without executing source on the host."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .coding import CodingEvaluation


@dataclass(frozen=True, slots=True)
class CodeQualityReport:
    syntax_valid: bool
    function_count: int
    branching_count: int
    line_count: int
    suggestions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodingFeedback:
    score: float
    hints: tuple[str, ...]
    test_explanations: tuple[str, ...]
    syntax_error: str | None
    quality: CodeQualityReport
    suggested_tests: tuple[str, ...]


class CodingFeedbackEngine:
    def analyze(self, source: str, evaluation: CodingEvaluation) -> CodingFeedback:
        syntax_error = None
        tree = None
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            syntax_error = f"Line {error.lineno}: {error.msg}"
        suggestions: list[str] = []
        if tree is not None:
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            branches = [node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While))]
            if not functions:
                suggestions.append("Consider placing the solution in a small named function.")
            if len(branches) > 4:
                suggestions.append("Consider splitting complex branching into smaller helpers.")
            quality = CodeQualityReport(True, len(functions), len(branches), len(source.splitlines()), tuple(suggestions))
            suggested_tests = tuple(f"assert {function.name}(...) == ..." for function in functions)
        else:
            quality = CodeQualityReport(False, 0, 0, len(source.splitlines()), ("Fix the syntax error before evaluating behavior.",))
            suggested_tests = ()
        hints = ("Read the first failing test carefully.", "Trace the inputs and outputs step by step.") if evaluation.tests_failed else ()
        explanations = tuple(
            f"Test {index} passed." if index <= evaluation.tests_passed else f"Test {index} failed; inspect its assertion and edge cases."
            for index in range(1, evaluation.tests_passed + evaluation.tests_failed + 1)
        )
        return CodingFeedback(evaluation.score, hints, explanations, syntax_error, quality, suggested_tests)
