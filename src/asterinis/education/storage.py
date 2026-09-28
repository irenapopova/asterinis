"""SQLite persistence for education profiles and progress events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from threading import Lock

from .models import LearnerProfile, ProgressEvent


class EducationStore:
    def __init__(self, path: str = "asterinis-education.db") -> None:
        self.path = path
        self._lock = Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS learner_profiles (
                learner_id TEXT NOT NULL,
                course_id TEXT NOT NULL,
                mastery TEXT NOT NULL,
                completed_lessons TEXT NOT NULL,
                preferences TEXT NOT NULL,
                PRIMARY KEY (learner_id, course_id)
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS progress_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                course_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                score REAL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._connection.commit()

    def save_profile(self, profile: LearnerProfile) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT INTO learner_profiles
                (learner_id, course_id, mastery, completed_lessons, preferences)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(learner_id, course_id) DO UPDATE SET
                mastery=excluded.mastery,
                completed_lessons=excluded.completed_lessons,
                preferences=excluded.preferences""",
                (
                    profile.learner_id,
                    profile.course_id,
                    json.dumps(profile.mastered_concepts),
                    json.dumps(sorted(profile.completed_lessons)),
                    json.dumps(profile.preferences, default=str),
                ),
            )
            self._connection.commit()

    def get_profile(self, learner_id: str, course_id: str) -> LearnerProfile | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT learner_id, course_id, mastery, completed_lessons, preferences "
                "FROM learner_profiles WHERE learner_id = ? AND course_id = ?",
                (learner_id, course_id),
            ).fetchone()
        if row is None:
            return None
        return LearnerProfile(
            learner_id=row[0],
            course_id=row[1],
            mastered_concepts=json.loads(row[2]),
            completed_lessons=set(json.loads(row[3])),
            preferences=json.loads(row[4]),
        )

    def add_event(self, event: ProgressEvent) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT INTO progress_events
                (learner_id, course_id, event_type, item_id, score, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.learner_id,
                    event.course_id,
                    event.event_type,
                    event.item_id,
                    event.score,
                    json.dumps(event.metadata, default=str),
                    event.created_at.isoformat(),
                ),
            )
            self._connection.commit()

    def events(self, learner_id: str, course_id: str) -> tuple[ProgressEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT learner_id, course_id, event_type, item_id, score, metadata, created_at "
                "FROM progress_events WHERE learner_id = ? AND course_id = ? ORDER BY id",
                (learner_id, course_id),
            ).fetchall()
        return tuple(
            ProgressEvent(
                learner_id=row[0],
                course_id=row[1],
                event_type=row[2],
                item_id=row[3],
                score=row[4],
                metadata=json.loads(row[5]),
                created_at=datetime.fromisoformat(row[6]),
            )
            for row in rows
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "EducationStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
