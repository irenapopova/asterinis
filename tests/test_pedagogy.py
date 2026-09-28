from asterinis.education import (
    CreativityEngine,
    HintLadder,
    LearnerProfile,
    TeachingMode,
    Tutor,
)


def test_pedagogy_adapts_to_mastery_and_attempts() -> None:
    tutor = Tutor()
    beginner = tutor.respond(
        LearnerProfile("student", "python"),
        concept="loops",
        question="Why does my loop fail?",
    )
    advanced = tutor.respond(
        LearnerProfile("student", "python", {"loops": 0.9}),
        concept="loops",
        question="Why does my loop fail?",
    )

    assert beginner.plan.mode == TeachingMode.SOCRATIC
    assert advanced.plan.mode == TeachingMode.CHALLENGE


def test_hints_and_creative_challenges_support_productive_struggle() -> None:
    ladder = HintLadder(("Look at the loop condition.", "Trace the values step by step."))
    assert ladder.next(0).level == 1
    assert ladder.next(2) is None

    challenge = CreativityEngine().create(
        "functions",
        kind="alternative",
        constraints=("do not use global variables",),
    )
    assert "two different ways" in challenge.prompt
    assert challenge.constraints == ("do not use global variables",)
