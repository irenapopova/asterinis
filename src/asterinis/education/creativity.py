"""Creative-thinking challenge generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreativeChallenge:
    concept: str
    prompt: str
    constraints: tuple[str, ...]
    thinking_skills: tuple[str, ...]


class CreativityEngine:
    def create(
        self,
        concept: str,
        *,
        constraints: tuple[str, ...] = (),
        kind: str = "alternative",
    ) -> CreativeChallenge:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept cannot be empty.")
        prompts = {
            "alternative": f"Solve a {concept} problem in two different ways and compare the tradeoffs.",
            "constraint": f"Create a solution using {concept} while respecting every constraint.",
            "prediction": f"Predict what will happen in a {concept} example before testing it.",
            "design": f"Design your own small project that demonstrates {concept} and explain your choices.",
            "what_if": f"Explore what would change in a {concept} solution if one important assumption changed.",
        }
        if kind not in prompts:
            raise ValueError(f"unknown creative challenge kind: {kind}")
        return CreativeChallenge(
            concept=concept,
            prompt=prompts[kind],
            constraints=tuple(item.strip() for item in constraints if item.strip()),
            thinking_skills=("divergent thinking", "reasoning", "reflection"),
        )
