"""Structured tutoring responses for local models, RAG, or LLM providers."""

from __future__ import annotations

from dataclasses import dataclass

from .creativity import CreativityEngine
from .hints import HintLadder
from .models import LearnerProfile
from .pedagogy import PedagogyEngine, TeachingPlan, TeachingRequest
from .reflections import ReflectionEngine


@dataclass(frozen=True, slots=True)
class TutorResponse:
    plan: TeachingPlan
    content_prompt: str
    hint: str | None = None
    reflection_questions: tuple[str, ...] = ()


class Tutor:
    """Create teaching instructions while leaving answer generation to a provider."""

    def __init__(self, pedagogy: PedagogyEngine | None = None) -> None:
        self.pedagogy = pedagogy or PedagogyEngine()
        self.creativity = CreativityEngine()
        self.reflections = ReflectionEngine()

    def respond(
        self,
        profile: LearnerProfile,
        *,
        concept: str,
        question: str,
        attempts: int = 0,
        student_attempt: str | None = None,
        hint_ladder: HintLadder | None = None,
    ) -> TutorResponse:
        request = TeachingRequest(
            learner_id=profile.learner_id,
            concept=concept,
            question=question,
            mastery=profile.mastered_concepts.get(concept, 0.0),
            attempts=attempts,
            student_attempt=student_attempt,
        )
        plan = self.pedagogy.choose(request)
        hint = hint_ladder.next(attempts).text if hint_ladder and plan.mode.value == "hint" and hint_ladder.next(attempts) else None
        reflection = self.reflections.create(concept).questions if plan.mode.value == "reflect" else ()
        return TutorResponse(
            plan=plan,
            content_prompt=(
                f"Learner question: {question}\n"
                f"Teaching instruction: {plan.prompt}\n"
                "Preserve the learner's opportunity to reason before revealing a solution."
            ),
            hint=hint,
            reflection_questions=reflection,
        )
