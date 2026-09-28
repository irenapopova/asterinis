"""Lightweight SM-2-style spaced repetition scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class ReviewSchedule:
    item_id: str
    repetitions: int
    interval_days: int
    ease_factor: float
    next_review: datetime


class SpacedRepetitionScheduler:
    def review(
        self,
        item_id: str,
        *,
        quality: int,
        schedule: ReviewSchedule | None = None,
        now: datetime | None = None,
    ) -> ReviewSchedule:
        if not item_id.strip():
            raise ValueError("item_id cannot be empty.")
        if not 0 <= quality <= 5:
            raise ValueError("quality must be between 0 and 5.")
        current = schedule or ReviewSchedule(item_id, 0, 0, 2.5, now or datetime.now(timezone.utc))
        if current.item_id != item_id:
            raise ValueError("schedule item does not match item_id.")
        timestamp = now or datetime.now(timezone.utc)
        ease = max(
            1.3,
            current.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
        )
        if quality < 3:
            repetitions = 0
            interval = 1
        else:
            repetitions = current.repetitions + 1
            interval = 1 if repetitions == 1 else 6 if repetitions == 2 else round(current.interval_days * ease)
            interval = max(1, interval)
        return ReviewSchedule(item_id, repetitions, interval, round(ease, 4), timestamp + timedelta(days=interval))
