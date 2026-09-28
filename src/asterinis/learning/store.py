from __future__ import annotations

from collections import defaultdict
from threading import Lock

from .records import StrategyRecord


class StrategyStore:
    """
    Thread-safe in-memory history of strategy outcomes.

    This store is intentionally lightweight. A persistent backend can be
    introduced later without changing the scoring and selection APIs.
    """

    def __init__(self) -> None:
        self._records: list[StrategyRecord] = []
        self._lock = Lock()

    def add(
        self,
        record: StrategyRecord,
    ) -> None:
        if not isinstance(record, StrategyRecord):
            raise TypeError(
                "record must be a StrategyRecord."
            )

        with self._lock:
            self._records.append(record)

    def extend(
        self,
        records: list[StrategyRecord],
    ) -> None:
        for record in records:
            if not isinstance(record, StrategyRecord):
                raise TypeError(
                    "Every record must be a StrategyRecord."
                )

        with self._lock:
            self._records.extend(records)

    def all(self) -> tuple[StrategyRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def for_query_type(
        self,
        query_type: str,
    ) -> tuple[StrategyRecord, ...]:
        query_type = query_type.strip()

        if not query_type:
            raise ValueError(
                "query_type cannot be empty."
            )

        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.query_type == query_type
            )

    def for_strategy(
        self,
        strategy: str,
    ) -> tuple[StrategyRecord, ...]:
        strategy = strategy.strip()

        if not strategy:
            raise ValueError(
                "strategy cannot be empty."
            )

        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.strategy == strategy
            )

    def grouped_by_strategy(
        self,
        *,
        query_type: str | None = None,
    ) -> dict[str, tuple[StrategyRecord, ...]]:
        grouped: dict[
            str,
            list[StrategyRecord],
        ] = defaultdict(list)

        with self._lock:
            for record in self._records:
                if (
                    query_type is not None
                    and record.query_type != query_type
                ):
                    continue

                grouped[record.strategy].append(record)

        return {
            strategy: tuple(records)
            for strategy, records in grouped.items()
        }

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)