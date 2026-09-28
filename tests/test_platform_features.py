from asterinis.education import (
    AntiCheatingAnalyzer,
    CertificateIssuer,
    CodeQualityReport,
    CodingFeedbackEngine,
    CourseEntitlement,
    EntitlementManager,
    LearningAnalytics,
    PairProgrammingSession,
    ProjectAssignment,
    ProjectMilestone,
    SandboxResult,
    build_dashboard,
)
from asterinis.education.coding import CodingEvaluation
from asterinis.education.models import ProgressEvent


def test_coding_feedback_supports_partial_credit() -> None:
    evaluation = CodingEvaluation("ex", False, 1, 1, SandboxResult(False, "", "", 1, 3.0), "")
    feedback = CodingFeedbackEngine().analyze("def add(a, b):\n    return a + b", evaluation)
    assert feedback.score == 0.5
    assert feedback.quality.syntax_valid


def test_projects_pairing_billing_certificates_and_anti_cheating() -> None:
    project = ProjectAssignment("p", "Project", "Build it", (ProjectMilestone("m", "M", "Do M"),))
    project.complete("m")
    assert project.completion == 1.0
    session = PairProgrammingSession("s", "ex", "a", "b")
    session.add_message("a", "I found the bug.")
    manager = EntitlementManager()
    manager.apply_webhook(CourseEntitlement("a", "python", "active", "customer"))
    assert manager.can_access("a", "python")
    assert CertificateIssuer().issue("a", "python", completion=1.0).course_id == "python"
    report = AntiCheatingAnalyzer().report("ex", {"a": "x = 1", "b": "x = 1"})
    assert report.matches


def test_analytics_builds_instructor_dashboard() -> None:
    events = (
        ProgressEvent("a", "python", "assessment", "q", score=0.2),
        ProgressEvent("a", "python", "lesson_completed", "l"),
    )
    summary = LearningAnalytics().summarize("a", events)
    dashboard = build_dashboard((summary,))
    assert dashboard.struggling_learners == ("a",)
