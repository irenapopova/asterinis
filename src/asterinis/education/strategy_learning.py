"""Contextual bandit for learning which teaching strategy works best."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True, slots=True)
class StrategySelection:
    strategy: str
    context: tuple[str, ...]
    exploration: bool


@dataclass(slots=True)
class _Arm:
    rewards: int = 0
    total_reward: float = 0.0


class TeachingStrategyBandit:
    """Epsilon-greedy contextual strategy selector.

    Rewards should represent learner outcomes, for example 1.0 for helpful
    feedback and 0.0 for unhelpful feedback. Statistics are kept per context.
    """

    def __init__(self, strategies: tuple[str, ...], *, exploration: float = 0.1, seed: int | None = None) -> None:
        if not strategies or any(not item.strip() for item in strategies):
            raise ValueError("strategies must contain non-empty names.")
        if len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be unique.")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be between 0 and 1.")
        self.strategies = strategies
        self.exploration = exploration
        self._random = random.Random(seed)
        self._arms: dict[tuple[tuple[str, ...], str], _Arm] = {}

    def choose(self, context: tuple[str, ...] = ()) -> StrategySelection:
        context = tuple(item.strip().lower() for item in context if item.strip())
        explore = self._random.random() < self.exploration
        if explore:
            strategy = self._random.choice(self.strategies)
        else:
            strategy = max(self.strategies, key=lambda item: self._value(context, item))
        return StrategySelection(strategy, context, explore)

    def record(self, selection: StrategySelection, *, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be between 0 and 1.")
        arm = self._arms.setdefault((selection.context, selection.strategy), _Arm())
        arm.rewards += 1
        arm.total_reward += reward

    def average_reward(self, strategy: str, context: tuple[str, ...] = ()) -> float:
        arm = self._arms.get((tuple(context), strategy))
        return arm.total_reward / arm.rewards if arm and arm.rewards else 0.0

    def _value(self, context: tuple[str, ...], strategy: str) -> float:
        arm = self._arms.get((context, strategy))
        if arm is None or arm.rewards == 0:
            return 0.5
        total = sum(item.rewards for (item_context, _), item in self._arms.items() if item_context == context)
        uncertainty = math.sqrt(math.log(max(total, 1)) / arm.rewards)
        return arm.total_reward / arm.rewards + uncertainty * 0.1
