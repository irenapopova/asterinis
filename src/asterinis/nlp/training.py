from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    epochs: int
    metrics: dict[str, float] = field(default_factory=dict)


class NLPTrainer(ABC):
    """Backend-neutral training contract for model adapters."""

    @abstractmethod
    def train(self, samples: Iterable[Any], *, config: TrainingConfig) -> TrainingResult:
        raise NotImplementedError
