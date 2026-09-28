from asterinis.education import (
    Assessment,
    AssessmentEngine,
    Question,
    QuestionType,
)


def test_assessment_engine_grades_questions_and_partial_score() -> None:
    assessment = Assessment(
        "python-basics",
        "Python Basics",
        questions=(
            Question(
                "q1",
                "What keyword defines a function?",
                "functions",
                QuestionType.MULTIPLE_CHOICE,
                answer="def",
                options=("def", "class", "loop"),
            ),
            Question(
                "q2",
                "Is Python dynamically typed?",
                "types",
                QuestionType.TRUE_FALSE,
                answer="true",
            ),
            Question(
                "q3",
                "Name a loop keyword.",
                "loops",
                QuestionType.SHORT_ANSWER,
                answer=("for", "while"),
                explanation="Python supports for and while loops.",
            ),
        ),
    )

    result = AssessmentEngine().grade(
        assessment,
        {"q1": "DEF", "q2": "false", "q3": "while"},
    )

    assert result.score == 2 / 3
    assert not result.passed
    assert result.questions[1].feedback != "Correct."


def test_assessment_requires_unique_question_ids() -> None:
    question = Question("q1", "Question", "concept", QuestionType.SHORT_ANSWER, "answer")

    try:
        Assessment("test", "Test", (question, question))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate question IDs should be rejected")
