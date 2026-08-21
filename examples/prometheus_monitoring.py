from asterinis.observability.prometheus import (
    PrometheusExporter,
)


def main() -> None:
    exporter = PrometheusExporter()

    exporter.record_request(
        route="rag",
        duration_seconds=0.125,
    )

    exporter.record_agent_run(
        agent="retrieval-quality",
    )

    exporter.record_retrieval(
        retriever="bm25",
        duration_seconds=0.042,
    )

    print(
        "Prometheus metrics recorded."
    )


if __name__ == "__main__":
    main()