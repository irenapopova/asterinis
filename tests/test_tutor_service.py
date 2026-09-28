from asterinis.education import (
    Course,
    CourseContext,
    LearnerProfile,
    Lesson,
    TutorService,
)


def test_tutor_service_grounds_answer_in_course_content() -> None:
    course = Course(
        "python",
        "Python Basics",
        lessons=(
            Lesson(
                "loops",
                "Loops",
                "A for loop repeats over each item in an iterable.",
                concepts=("loops",),
            ),
        ),
    )
    captured: dict[str, str] = {}

    def provider(prompt: str, context: str) -> str:
        captured["prompt"] = prompt
        captured["context"] = context
        return "A course-grounded answer."

    service = TutorService(CourseContext.from_course(course), provider)
    result = service.answer(
        LearnerProfile("student-1", "python"),
        concept="loops",
        question="How does a for loop work?",
    )

    assert result.answer == "A course-grounded answer."
    assert "for loop repeats" in captured["context"]
    assert result.citations[0]["document_id"] == "loops"
    assert result.next_lesson_id == "loops"
