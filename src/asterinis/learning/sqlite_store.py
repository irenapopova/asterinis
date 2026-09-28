from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from threading import Lock

from .records import StrategyRecord


class SQLiteStrategyStore:
    """Persistent SQLite implementation of the strategy history store."""

    def __init__(self, path: str = "asterinis-learning.db") -> None:
        self.path = path
        self._lock = Lock()
        self._connection = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                query_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality_score REAL,
                latency_seconds REAL,
                cost REAL,
                confidence REAL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def add(self, record: StrategyRecord) -> None:
        self._validate_record(record)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO strategy_records (
                    strategy, query_type, success, quality_score,
                    latency_seconds, cost, confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.strategy,
                    record.query_type,
                    int(record.success),
                    record.quality_score,
                    record.latency_seconds,
                    record.cost,
                    record.confidence,
                    json.dumps(record.metadata, default=str),
                    record.created_at.isoformat(),
                ),
            )
            self._connection.commit()

    def extend(self, records: list[StrategyRecord]) -> None:
        for record in records:
            self._validate_record(record)

        with self._lock:
            self._connection.executemany(
                """
                INSERT INTO strategy_records (
                    strategy, query_type, success, quality_score,
                    latency_seconds, cost, confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.strategy,
                        record.query_type,
                        int(record.success),
                        record.quality_score,
                        record.latency_seconds,
                        record.cost,
                        record.confidence,
                        json.dumps(record.metadata, default=str),
                        record.created_at.isoformat(),
                    )
                    for record in records
                ],
            )
            self._connection.commit()

    def all(self) -> tuple[StrategyRecord, ...]:
        return self._query("SELECT * FROM strategy_records ORDER BY id")

    def for_query_type(self, query_type: str) -> tuple[StrategyRecord, ...]:
        query_type = query_type.strip()
        if not query_type:
            raise ValueError("query_type cannot be empty.")
        return self._query(
            "SELECT * FROM strategy_records WHERE query_type = ? ORDER BY id",
            (query_type,),
        )

    def for_strategy(self, strategy: str) -> tuple[StrategyRecord, ...]:
        strategy = strategy.strip()
        if not strategy:
            raise ValueError("strategy cannot be empty.")
        return self._query(
            "SELECT * FROM strategy_records WHERE strategy = ? ORDER BY id",
            (strategy,),
        )

    def grouped_by_strategy(
        self,
        *,
        query_type: str | None = None,
    ) -> dict[str, tuple[StrategyRecord, ...]]:
        records = (
            self.for_query_type(query_type)
            if query_type is not None
            else self.all()
        )
        grouped: dict[str, list[StrategyRecord]] = defaultdict(list)
        for record in records:
            grouped[record.strategy].append(record)
        return {
            strategy: tuple(items)
            for strategy, items in grouped.items()
        }

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM strategy_records")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM strategy_records"
            ).fetchone()
        return int(row[0])

    def _query(
        self,
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> tuple[StrategyRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                statement,
                parameters,
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    @staticmethod
    def _validate_record(record: StrategyRecord) -> None:
        if not isinstance(record, StrategyRecord):
            raise TypeError("record must be a StrategyRecord.")

    @staticmethod
    def _record_from_row(row: tuple[object, ...]) -> StrategyRecord:
        return StrategyRecord(
            strategy=str(row[1]),
            query_type=str(row[2]),
            success=bool(row[3]),
            quality_score=row[4],
            latency_seconds=row[5],
            cost=row[6],
            confidence=row[7],
            metadata=json.loads(str(row[8])),
            created_at=datetime.fromisoformat(str(row[9])),
        )
