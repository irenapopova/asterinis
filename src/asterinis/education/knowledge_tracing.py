"""Bayesian Knowledge Tracing for concept mastery estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeState:
    concept: str
    probability_known: float
    observations: int


class BayesianKnowledgeTracer:
    def __init__(
        self,
        *,
        prior: float = 0.2,
        learn_probability: float = 0.2,
        guess_probability: float = 0.2,
        slip_probability: float = 0.1,
    ) -> None:
        values = (prior, learn_probability, guess_probability, slip_probability)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("BKT probabilities must be between 0 and 1.")
        self.prior = prior
        self.learn_probability = learn_probability
        self.guess_probability = guess_probability
        self.slip_probability = slip_probability
        self._states: dict[str, KnowledgeState] = {}

    def observe(self, concept: str, *, correct: bool) -> KnowledgeState:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept cannot be empty.")
        previous = self._states.get(concept, KnowledgeState(concept, self.prior, 0))
        if correct:
            likelihood_known = 1.0 - self.slip_probability
            likelihood_unknown = self.guess_probability
        else:
            likelihood_known = self.slip_probability
            likelihood_unknown = 1.0 - self.guess_probability
        denominator = previous.probability_known * likelihood_known + (1 - previous.probability_known) * likelihood_unknown
        posterior = previous.probability_known if denominator == 0 else previous.probability_known * likelihood_known / denominator
        updated = posterior + (1 - posterior) * self.learn_probability
        state = KnowledgeState(concept, min(1.0, max(0.0, updated)), previous.observations + 1)
        self._states[concept] = state
        return state

    def state(self, concept: str) -> KnowledgeState:
        concept = concept.strip()
        return self._states.get(concept, KnowledgeState(concept, self.prior, 0))

    def all(self) -> tuple[KnowledgeState, ...]:
        return tuple(self._states.values())
