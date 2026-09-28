"""Reflection prompts for durable learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReflectionPrompt:
    concept: str
    questions: tuple[str, ...]


class ReflectionEngine:
    def create(self, concept: str, *, after_success: bool = False) -> ReflectionPrompt:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept cannot be empty.")
        questions = (
            f"What was the key idea behind {concept}?",
            f"What mistake or assumption would be easiest to make with {concept}?",
            f"Where could you use {concept} in a new situation?",
        )
        if after_success:
            questions = questions + (f"How could you make your {concept} solution clearer or more efficient?",)
        return ReflectionPrompt(concept, questions)
