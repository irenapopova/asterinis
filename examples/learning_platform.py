"""Run: ./.venv/bin/python examples/learning_platform.py"""

from asterinis import LearningPlatform
from asterinis.decision import DecisionPolicy, ProviderOption


platform = LearningPlatform("learning-demo.db")
platform.register_provider(
    ProviderOption(
        name="local-tutor",
        capability="generation",
        handler=lambda question: f"Local tutor answer: {question}",
        quality=0.75,
        cost=0.0,
        latency_ms=30,
    )
)

response = platform.ask(
    "Explain photosynthesis to a beginner.",
    policy=DecisionPolicy(mode="local"),
)
print(response.answer)
print("provider:", response.decision.provider)
print("response_id:", response.response_id)

# A real website would call this after the learner clicks Helpful/Not helpful.
platform.record_feedback(
    response.response_id,
    helpful=True,
    quality_score=1.0,
)
print("stored executions and feedback:", len(platform.store))
platform.close()
