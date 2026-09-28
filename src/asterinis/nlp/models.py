from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import NLPProvider


@dataclass(slots=True)
class ModelCard:
    name: str
    task: str
    languages: set[str] = field(default_factory=set)
    description: str = ""
    source: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.task = self.task.strip().lower()
        if not self.name or not self.task:
            raise ValueError("model name and task cannot be empty.")
        self.languages = {
            language.strip().lower()
            for language in self.languages
            if language.strip()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "languages": sorted(self.languages),
            "description": self.description,
            "source": self.source,
            "license": self.license,
            "metadata": dict(self.metadata),
        }


ModelFactory = Callable[[], NLPProvider]


class NLPModelRegistry:
    """Named model factories with lazy provider initialization."""

    def __init__(self) -> None:
        self._factories: dict[str, tuple[ModelCard, ModelFactory]] = {}
        self._loaded: dict[str, NLPProvider] = {}

    def register(self, card: ModelCard, factory: ModelFactory, *, replace: bool = False) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable.")
        if card.name in self._factories and not replace:
            raise ValueError(f"Model '{card.name}' is already registered.")
        self._factories[card.name] = (card, factory)
        self._loaded.pop(card.name, None)

    def card(self, name: str) -> ModelCard:
        try:
            return self._factories[name][0]
        except KeyError as exc:
            raise KeyError(f"Model '{name}' is not registered.") from exc

    def load(self, name: str) -> NLPProvider:
        if name not in self._loaded:
            try:
                provider = self._factories[name][1]()
            except KeyError as exc:
                raise KeyError(f"Model '{name}' is not registered.") from exc
            if not isinstance(provider, NLPProvider):
                raise TypeError("model factory must return an NLPProvider.")
            self._loaded[name] = provider
        return self._loaded[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def unload(self, name: str) -> None:
        self._loaded.pop(name, None)


def normalize_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return float(value)
