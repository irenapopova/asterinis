"""Provider-independent teaching strategy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TeachingMode(str, Enum):
    SOCRATIC = "socratic"
    HINT = "hint"
    EXPLAIN = "explain"
    CHALLENGE = "challenge"
    REFLECT = "reflect"
    DEBUG = "debug"
    COMPARE = "compare"


@dataclass(frozen=True, slots=True)
class TeachingRequest:
    learner_id: str
    concept: str
    question: str
    mastery: float = 0.0
    attempts: int = 0
    requested_mode: TeachingMode | None = None
    student_attempt: str | None = None

    def __post_init__(self) -> None:
        if not self.learner_id.strip() or not self.concept.strip() or not self.question.strip():
            raise ValueError("learner_id, concept, and question cannot be empty.")
        if not 0.0 <= self.mastery <= 1.0 or self.attempts < 0:
            raise ValueError("mastery must be between 0 and 1 and attempts cannot be negative.")


@dataclass(frozen=True, slots=True)
class TeachingPlan:
    mode: TeachingMode
    objective: str
    prompt: str
    next_modes: tuple[TeachingMode, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


class PedagogyEngine:
    """Select teaching behavior without generating model-specific text."""

    def choose(self, request: TeachingRequest) -> TeachingPlan:
        if request.requested_mode is not None:
            mode = request.requested_mode
        elif request.student_attempt and request.attempts > 0:
            mode = TeachingMode.DEBUG
        elif request.mastery < 0.35:
            mode = TeachingMode.SOCRATIC
        elif request.mastery < 0.7:
            mode = TeachingMode.HINT
        else:
            mode = TeachingMode.CHALLENGE

        prompts = {
            TeachingMode.SOCRATIC: f"Ask a guiding question about {request.concept}; do not reveal the answer yet.",
            TeachingMode.HINT: f"Give one small hint for {request.concept}, then ask the learner to try again.",
            TeachingMode.EXPLAIN: f"Explain {request.concept} clearly, then check understanding with one question.",
            TeachingMode.CHALLENGE: f"Give a harder {request.concept} problem with at least two possible approaches.",
            TeachingMode.REFLECT: f"Ask the learner to explain what they tried and what they would change about {request.concept}.",
            TeachingMode.DEBUG: f"Inspect the learner's attempt for a misconception about {request.concept}; ask a debugging question before correcting it.",
            TeachingMode.COMPARE: f"Ask the learner to compare two solutions for {request.concept} and discuss their tradeoffs.",
        }
        next_modes = {
            TeachingMode.SOCRATIC: (TeachingMode.HINT, TeachingMode.EXPLAIN),
            TeachingMode.HINT: (TeachingMode.CHALLENGE, TeachingMode.EXPLAIN),
            TeachingMode.EXPLAIN: (TeachingMode.SOCRATIC, TeachingMode.CHALLENGE),
            TeachingMode.CHALLENGE: (TeachingMode.REFLECT, TeachingMode.COMPARE),
            TeachingMode.REFLECT: (TeachingMode.CHALLENGE,),
            TeachingMode.DEBUG: (TeachingMode.HINT, TeachingMode.EXPLAIN),
            TeachingMode.COMPARE: (TeachingMode.REFLECT, TeachingMode.CHALLENGE),
        }[mode]
        return TeachingPlan(
            mode=mode,
            objective=f"Help the learner develop understanding of {request.concept}.",
            prompt=prompts[mode],
            next_modes=next_modes,
            metadata={"mastery": str(request.mastery), "attempts": str(request.attempts)},
        )
