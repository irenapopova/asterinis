from asterinis.education import (
    CodingEvaluator,
    CodingExercise,
    SandboxResult,
)
from asterinis.education.sandbox import CodeSandbox


class FakeSandbox(CodeSandbox):
    def __init__(self, result: SandboxResult) -> None:
        self.result = result
        self.source = ""

    def run(self, source: str, *, stdin: str = "") -> SandboxResult:
        self.source = source
        return self.result


def test_coding_evaluator_reports_passed_tests() -> None:
    sandbox = FakeSandbox(
        SandboxResult(True, "ASTERINIS_TESTS:2:0", "", 0, 10.0)
    )
    exercise = CodingExercise(
        "add",
        "Add numbers",
        "Implement add.",
        ("assert add(1, 2) == 3", "assert add(0, 0) == 0"),
        "functions",
    )

    result = CodingEvaluator(sandbox).evaluate(exercise, "def add(a, b): return a + b")

    assert result.passed
    assert result.tests_passed == 2
    assert "assert add" in sandbox.source


def test_sandbox_limits_are_validated() -> None:
    from asterinis.education import SandboxLimits

    try:
        SandboxLimits(timeout_seconds=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid sandbox limits should be rejected")
