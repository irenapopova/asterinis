"""Coding exercise definitions and sandbox-backed evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .sandbox import CodeSandbox, SandboxResult


@dataclass(frozen=True, slots=True)
class CodingExercise:
    exercise_id: str
    title: str
    prompt: str
    tests: tuple[str, ...]
    concept: str
    starter_code: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.exercise_id.strip() or not self.title.strip() or not self.prompt.strip() or not self.concept.strip():
            raise ValueError("exercise fields cannot be empty.")
        if not self.tests or any(not test.strip() for test in self.tests):
            raise ValueError("exercise must contain tests.")


@dataclass(frozen=True, slots=True)
class CodingEvaluation:
    exercise_id: str
    passed: bool
    tests_passed: int
    tests_failed: int
    sandbox: SandboxResult
    feedback: str

    @property
    def score(self) -> float:
        total = self.tests_passed + self.tests_failed
        return self.tests_passed / total if total else 0.0


class CodingEvaluator:
    def __init__(self, sandbox: CodeSandbox) -> None:
        self.sandbox = sandbox

    def evaluate(self, exercise: CodingExercise, source: str) -> CodingEvaluation:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string.")
        program = self._build_program(source, exercise.tests)
        result = self.sandbox.run(program)
        passed, failed = self._parse_counts(result.stdout)
        return CodingEvaluation(
            exercise.exercise_id,
            result.passed and failed == 0,
            passed,
            failed,
            result,
            self._feedback(result, passed, failed),
        )

    @staticmethod
    def _build_program(source: str, tests: tuple[str, ...]) -> str:
        blocks = []
        for test in tests:
            indented = test.replace("\n", "\n    ")
            blocks.append(f"try:\n    {indented}\n    passed += 1\nexcept Exception:\n    failed += 1\n    import traceback\n    traceback.print_exc()")
        return f"{source}\n\npassed = 0\nfailed = 0\n" + "\n".join(blocks) + "\nprint('ASTERINIS_TESTS:' + str(passed) + ':' + str(failed))\n"

    @staticmethod
    def _parse_counts(stdout: str) -> tuple[int, int]:
        match = re.search(r"ASTERINIS_TESTS:(\d+):(\d+)", stdout)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 1)

    @staticmethod
    def _feedback(result: SandboxResult, passed: int, failed: int) -> str:
        if result.timed_out:
            return "Your program exceeded the time limit. Check for an infinite loop or an inefficient algorithm."
        if result.exit_code not in {0, None} and failed == 0:
            return "Your program stopped before the tests completed. Check the syntax and error output."
        if failed:
            return f"{passed} test(s) passed and {failed} failed. Use the failing test output to debug your solution."
        return "All tests passed. Explain why your solution works and consider another approach."


__all__ = ["CodingEvaluation", "CodingEvaluator", "CodingExercise"]
