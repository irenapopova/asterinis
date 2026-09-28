from asterinis.education import (
    Assessment,
    EducationService,
    EducationStore,
    LearnerProfile,
    Question,
    QuestionType,
)


def test_service_grades_and_updates_mastery(tmp_path) -> None:
    store = EducationStore(str(tmp_path / "education.db"))
    service = EducationService(store)
    profile = LearnerProfile("student-1", "python")
    assessment = Assessment(
        "functions-check",
        "Functions Check",
        questions=(
            Question(
                "q1",
                "What keyword defines a function?",
                "functions",
                QuestionType.MULTIPLE_CHOICE,
                answer="def",
                options=("def", "class"),
            ),
        ),
    )

    result = service.submit_assessment(
        profile,
        assessment,
        {"q1": "def"},
        lesson_id="functions-lesson",
    )

    assert result.passed
    assert profile.mastered_concepts["functions"] == 0.3
    assert profile.completed_lessons == {"functions-lesson"}
    assert len(store.events("student-1", "python")) == 2
    service.close()
