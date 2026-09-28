"""Optional training bridge for Flair models.

This module intentionally imports Flair only when a method is used. Asterinis
therefore remains installable without PyTorch or Flair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asterinis.nlp.training import (
    TrainingCorpus,
    TrainingResult,
    TrainingRunConfig,
    TrainingTask,
)


@dataclass(slots=True)
class FlairTrainingConfig:
    """Common options for Flair training and transformer fine-tuning."""

    output_path: str
    mode: str = "train"
    epochs: int = 10
    learning_rate: float = 0.1
    batch_size: int = 32
    mini_batch_chunk_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_path = self.output_path.strip()
        if not self.output_path:
            raise ValueError("output_path cannot be empty.")
        if self.mode not in {"train", "fine_tune", "custom"}:
            raise ValueError("mode must be train, fine_tune, or custom.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.mini_batch_chunk_size is not None and self.mini_batch_chunk_size < 1:
            raise ValueError("mini_batch_chunk_size must be positive.")


class FlairTrainerAdapter:
    """Asterinis adapter for Flair corpus/model training.

    The adapter supports Flair's classic training, transformer fine-tuning,
    and custom training modes. Model architecture and corpus construction stay
    in Flair, while Asterinis owns the stable application-facing interface.
    """

    @staticmethod
    def _trainer(model: Any, corpus: Any) -> Any:
        try:
            from flair.trainers import ModelTrainer
        except ImportError as exc:
            raise ImportError(
                'Flair training requires: pip install "asterinis[flair]"'
            ) from exc
        return ModelTrainer(model, corpus)

    def train(
        self,
        model: Any,
        corpus: Any,
        *,
        config: FlairTrainingConfig,
    ) -> TrainingResult:
        trainer = self._trainer(model, corpus)
        options = {
            "learning_rate": config.learning_rate,
            "mini_batch_size": config.batch_size,
            "max_epochs": config.epochs,
            **config.extra,
        }
        if config.mini_batch_chunk_size is not None:
            options["mini_batch_chunk_size"] = config.mini_batch_chunk_size

        if config.mode == "fine_tune":
            trainer.fine_tune(config.output_path, **options)
        elif config.mode == "custom":
            trainer.train_custom(config.output_path, **options)
        else:
            trainer.train(config.output_path, **options)

        return TrainingResult(
            model_name=config.output_path,
            epochs=config.epochs,
            metrics={},
        )

    def train_asterinis(
        self,
        corpus: TrainingCorpus,
        *,
        config: TrainingRunConfig,
        model: Any,
        flair_corpus: Any,
    ) -> TrainingResult:
        """Train any Flair model using an Asterinis-owned run configuration.

        ``flair_corpus`` is supplied by the integration boundary; the rest of
        the application only needs ``TrainingCorpus`` and ``TrainingRunConfig``.
        All supported Flair task types use the same training lifecycle.
        """
        if not isinstance(config.task, TrainingTask):
            raise TypeError("config.task must be a TrainingTask.")
        if not corpus.train:
            raise ValueError("corpus must contain training samples.")
        return self.train(
            model,
            flair_corpus,
            config=FlairTrainingConfig(
                output_path=config.output_path,
                mode="fine_tune" if config.fine_tune else "train",
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                extra=config.options,
            ),
        )

    @staticmethod
    def label_dictionary(corpus: Any, label_type: str, *, add_unk: bool = True) -> Any:
        """Create Flair's label dictionary from a corpus."""
        return corpus.make_label_dictionary(
            label_type=label_type,
            add_unk=add_unk,
        )

    @staticmethod
    def load_model(path: str) -> Any:
        """Load a trained Flair model for prediction."""
        try:
            from flair.nn import Classifier
        except ImportError as exc:
            raise ImportError(
                'Flair model loading requires: pip install "asterinis[flair]"'
            ) from exc
        return Classifier.load(path)


__all__ = ["FlairTrainerAdapter", "FlairTrainingConfig"]
