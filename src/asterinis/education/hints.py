"""Progressive hint ladders that preserve productive struggle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Hint:
    level: int
    text: str


class HintLadder:
    def __init__(self, hints: tuple[str, ...]) -> None:
        if not hints or any(not hint.strip() for hint in hints):
            raise ValueError("hints must contain non-empty text.")
        self._hints = tuple(Hint(index + 1, hint.strip()) for index, hint in enumerate(hints))

    def next(self, revealed_levels: int = 0) -> Hint | None:
        if revealed_levels < 0:
            raise ValueError("revealed_levels cannot be negative.")
        return self._hints[revealed_levels] if revealed_levels < len(self._hints) else None

    def all(self) -> tuple[Hint, ...]:
        return self._hints
