from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol


class TrainingTask(str, Enum):
    """Tasks exposed by Asterinis, independent of a model framework."""

    SEQUENCE_TAGGING = "sequence_tagging"
    TEXT_CLASSIFICATION = "text_classification"
    SPAN_CLASSIFICATION = "span_classification"
    RELATION_EXTRACTION = "relation_extraction"
    MULTITASK = "multitask"


@dataclass(frozen=True, slots=True)
class Annotation:
    """A framework-neutral labeled span or relation."""

    label: str
    start: int | None = None
    end: int | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("annotation label cannot be empty.")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together.")
        if self.start is not None and (self.start < 0 or self.end <= self.start):
            raise ValueError("annotation span must have a valid start and end.")


@dataclass(frozen=True, slots=True)
class TrainingSample:
    text: str
    annotations: tuple[Annotation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("training sample text cannot be empty.")


@dataclass(slots=True)
class TrainingCorpus:
    """Train/dev/test data owned by Asterinis rather than a backend."""

    train: tuple[TrainingSample, ...]
    dev: tuple[TrainingSample, ...] = ()
    test: tuple[TrainingSample, ...] = ()

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("training corpus must contain training samples.")

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({annotation.label for sample in self.all() for annotation in sample.annotations}))

    def all(self) -> tuple[TrainingSample, ...]:
        return self.train + self.dev + self.test


@dataclass(slots=True)
class TrainingRunConfig:
    task: TrainingTask
    output_path: str
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 8
    fine_tune: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_path = self.output_path.strip()
        if not self.output_path:
            raise ValueError("output_path cannot be empty.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


class TrainingBackend(Protocol):
    def train(
        self,
        corpus: TrainingCorpus,
        *,
        config: TrainingRunConfig,
    ) -> TrainingResult:
        ...


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
