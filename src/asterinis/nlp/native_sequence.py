"""Asterinis-native BiLSTM sequence tagger.

PyTorch is imported lazily so the base Asterinis installation stays small.
Annotations use character offsets and are converted to BIO token labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .training import TrainingCorpus, TrainingResult, TrainingRunConfig


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenPrediction:
    token: str
    label: str
    confidence: float
    start: int
    end: int


@dataclass(slots=True)
class NativeSequenceTaggerConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 13

    def __post_init__(self) -> None:
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1.")


def _torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn.utils.rnn import pad_sequence
    except ImportError as exc:
        raise ImportError(
            'Native neural training requires: pip install "asterinis[neural]"'
        ) from exc
    return torch, nn, pad_sequence


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in _TOKEN_PATTERN.finditer(text)]


def _bio_labels(
    token_spans: list[tuple[str, int, int]],
    annotations: tuple[Any, ...],
) -> list[str]:
    labels = ["O"] * len(token_spans)
    for annotation in annotations:
        if annotation.start is None or annotation.end is None:
            continue
        indexes = [
            index
            for index, (_, start, end) in enumerate(token_spans)
            if start < annotation.end and end > annotation.start
        ]
        if not indexes:
            continue
        for position, index in enumerate(indexes):
            prefix = "B-" if position == 0 else "I-"
            if labels[index] != "O":
                raise ValueError("training annotations overlap.")
            labels[index] = prefix + annotation.label
    return labels


class _BiLSTMTagger:
    def __init__(self, torch: Any, nn: Any, vocab_size: int, label_count: int, config: NativeSequenceTaggerConfig) -> None:
        self.module = nn.Sequential()  # replaced below; keeps the object easy to inspect
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * 2, label_count)
        self._torch = torch

    def parameters(self):
        return list(self.embedding.parameters()) + list(self.encoder.parameters()) + list(self.dropout.parameters()) + list(self.classifier.parameters())

    def state_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.state_dict(),
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.embedding.load_state_dict(state["embedding"])
        self.encoder.load_state_dict(state["encoder"])
        self.classifier.load_state_dict(state["classifier"])

    def train(self) -> None:
        self.embedding.train()
        self.encoder.train()
        self.dropout.train()
        self.classifier.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.encoder.eval()
        self.dropout.eval()
        self.classifier.eval()

    def __call__(self, inputs: Any) -> Any:
        embedded = self.embedding(inputs)
        encoded, _ = self.encoder(embedded)
        return self.classifier(self.dropout(encoded))


class NativeSequenceTagger:
    """A trainable Asterinis-native BiLSTM token classifier."""

    def __init__(self, *, config: NativeSequenceTaggerConfig | None = None) -> None:
        self.config = config or NativeSequenceTaggerConfig()
        self.token_to_id: dict[str, int] = {}
        self.label_to_id: dict[str, int] = {}
        self._model: _BiLSTMTagger | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_to_id)

    def fit(
        self,
        corpus: TrainingCorpus,
        *,
        config: TrainingRunConfig,
    ) -> TrainingResult:
        if not corpus.train:
            raise ValueError("corpus must contain training samples.")
        if config.task.value != "sequence_tagging":
            raise ValueError("NativeSequenceTagger requires sequence_tagging task.")
        torch, nn, pad_sequence = _torch()
        torch.manual_seed(self.config.seed)
        prepared = self._prepare(corpus)
        self._model = _BiLSTMTagger(torch, nn, len(self.token_to_id), len(self.label_to_id), self.config)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        losses: list[float] = []

        for _ in range(config.epochs):
            self._model.train()
            epoch_loss = 0.0
            for start in range(0, len(prepared), config.batch_size):
                batch = prepared[start : start + config.batch_size]
                inputs = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
                targets = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
                optimizer.zero_grad()
                output = self._model(inputs)
                loss = loss_function(output.reshape(-1, len(self.label_to_id)), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item())
            losses.append(epoch_loss)

        metrics = {"train_loss": losses[-1] if losses else 0.0}
        if corpus.dev:
            metrics.update({f"dev_{key}": value for key, value in self.evaluate(corpus.dev).items()})
        self.save(config.output_path)
        return TrainingResult(model_name=config.output_path, epochs=config.epochs, metrics=metrics)

    def predict(self, text: str) -> tuple[TokenPrediction, ...]:
        if self._model is None:
            raise RuntimeError("model is not trained or loaded.")
        torch, _, _ = _torch()
        token_spans = _tokens(text)
        if not token_spans:
            return ()
        self._model.eval()
        ids = torch.tensor([[self.token_to_id.get(token.lower(), 1) for token, _, _ in token_spans]], dtype=torch.long)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(ids), dim=-1)[0]
        predictions = probabilities.argmax(dim=-1)
        id_to_label = {value: key for key, value in self.label_to_id.items()}
        return tuple(
            TokenPrediction(token, id_to_label[int(label_id)], float(probabilities[index, label_id]), start, end)
            for index, (token, start, end) in enumerate(token_spans)
            for label_id in [predictions[index]]
        )

    def evaluate(self, samples: Any) -> dict[str, float]:
        total = correct = 0
        for sample in samples:
            expected = _bio_labels(_tokens(sample.text), sample.annotations)
            predicted = [item.label for item in self.predict(sample.text)]
            for actual, guess in zip(expected, predicted):
                total += 1
                correct += actual == guess
        return {"token_accuracy": correct / total if total else 0.0}

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("cannot save an untrained model.")
        torch, _, _ = _torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "token_to_id": self.token_to_id,
                "label_to_id": self.label_to_id,
                "state": self._model.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str) -> "NativeSequenceTagger":
        torch, _, _ = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tagger = cls(config=NativeSequenceTaggerConfig(**payload["config"]))
        tagger.token_to_id = payload["token_to_id"]
        tagger.label_to_id = payload["label_to_id"]
        tagger._model = _BiLSTMTagger(torch, __import__("torch").nn, len(tagger.token_to_id), len(tagger.label_to_id), tagger.config)
        tagger._model.load_state_dict(payload["state"])
        return tagger

    def _prepare(self, corpus: TrainingCorpus) -> list[tuple[Any, Any]]:
        torch, _, _ = _torch()
        samples = corpus.train
        words = {token.lower() for sample in samples for token, _, _ in _tokens(sample.text)}
        labels = {"O"}
        for sample in samples:
            labels.update(_bio_labels(_tokens(sample.text), sample.annotations))
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.token_to_id.update({word: index for index, word in enumerate(sorted(words), start=2)})
        self.label_to_id = {"<PAD>": 0}
        self.label_to_id.update({label: index for index, label in enumerate(sorted(labels), start=1)})
        prepared = []
        for sample in samples:
            token_spans = _tokens(sample.text)
            prepared.append(
                (
                    torch.tensor([self.token_to_id[token.lower()] for token, _, _ in token_spans], dtype=torch.long),
                    torch.tensor([self.label_to_id[label] for label in _bio_labels(token_spans, sample.annotations)], dtype=torch.long),
                )
            )
        return prepared


__all__ = ["NativeSequenceTagger", "NativeSequenceTaggerConfig", "TokenPrediction"]
