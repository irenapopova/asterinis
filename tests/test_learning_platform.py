from asterinis.decision import DecisionPolicy, ProviderOption
from asterinis.learning_platform import LearningPlatform


def test_learning_platform_routes_and_learns_feedback(tmp_path) -> None:
    platform = LearningPlatform(str(tmp_path / "learning.db"))
    platform.register_provider(
        ProviderOption(
            name="local-tutor",
            capability="generation",
            handler=lambda question: f"local: {question}",
            quality=0.75,
            cost=0.0,
            latency_ms=20,
        )
    )

    response = platform.ask(
        "Explain fractions",
        policy=DecisionPolicy(mode="local"),
    )

    assert response.answer == "local: Explain fractions"
    assert response.decision.provider == "local-tutor"

    platform.record_feedback(
        response.response_id,
        helpful=True,
        quality_score=0.9,
    )

    assert len(platform.store) == 2
    platform.close()


def test_feedback_rejects_unknown_response(tmp_path) -> None:
    platform = LearningPlatform(str(tmp_path / "learning.db"))

    try:
        platform.record_feedback("missing", helpful=False)
    except ValueError as error:
        assert "unknown response_id" in str(error)
    else:
        raise AssertionError("feedback should reject an unknown response")
    finally:
        platform.close()
