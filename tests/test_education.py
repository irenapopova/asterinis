from asterinis.education import (
    AssessmentResult,
    Course,
    EducationStore,
    LearnerProfile,
    Lesson,
    MasteryTracker,
    ProgressTracker,
)


def test_mastery_recommends_unfinished_weak_lesson() -> None:
    course = Course(
        "python",
        "Python Basics",
        lessons=(
            Lesson("variables", "Variables", "...", concepts=("variables",)),
            Lesson("loops", "Loops", "...", concepts=("loops",), prerequisites=("variables",)),
        ),
    )
    profile = LearnerProfile("learner-1", "python", {"variables": 0.9})

    assert MasteryTracker().next_lesson(profile, course).lesson_id == "loops"


def test_progress_persists_profile_and_events(tmp_path) -> None:
    path = str(tmp_path / "education.db")
    store = EducationStore(path)
    profile = LearnerProfile("learner-1", "python")
    tracker = ProgressTracker(store)

    score = tracker.record_assessment(
        profile,
        AssessmentResult("learner-1", "loops", 1.0, True, lesson_id="loops-1"),
    )
    assert score == 0.3
    assert profile.completed_lessons == {"loops-1"}
    assert len(store.events("learner-1", "python")) == 1
    store.close()

    reopened = EducationStore(path)
    loaded = reopened.get_profile("learner-1", "python")
    assert loaded is not None
    assert loaded.mastered_concepts["loops"] == 0.3
    assert loaded.completed_lessons == {"loops-1"}
    reopened.close()
