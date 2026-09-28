from asterinis.learning import (
    SQLiteStrategyStore,
    StrategyRecord,
)


def test_sqlite_strategy_store_persists_records(tmp_path) -> None:
    path = tmp_path / "learning.db"
    record = StrategyRecord(
        strategy="local-classifier",
        query_type="classification",
        success=True,
        quality_score=0.95,
        latency_seconds=0.02,
        cost=0.0,
        confidence=0.9,
        metadata={"source": "test"},
    )

    first_store = SQLiteStrategyStore(str(path))
    first_store.add(record)
    first_store.close()

    second_store = SQLiteStrategyStore(str(path))
    records = second_store.for_query_type("classification")

    assert len(records) == 1
    assert records[0].strategy == "local-classifier"
    assert records[0].metadata["source"] == "test"
    assert records[0].quality_score == 0.95
    second_store.close()


def test_sqlite_strategy_store_groups_records(tmp_path) -> None:
    store = SQLiteStrategyStore(str(tmp_path / "learning.db"))
    store.extend(
        [
            StrategyRecord("local", "classification", True),
            StrategyRecord("flair", "classification", True),
            StrategyRecord("local", "ner", True),
        ]
    )

    grouped = store.grouped_by_strategy(query_type="classification")

    assert set(grouped) == {"local", "flair"}
    assert len(store) == 3
    store.close()
