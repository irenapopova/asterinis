from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrometheusMetrics:
    requests_total: object
    request_duration_seconds: object
    provider_errors_total: object
    agent_runs_total: object
    retrieval_duration_seconds: object


class PrometheusExporter:
    """
    Optional Prometheus integration for Asterinis.

    Importing the observability package does not require prometheus-client.
    The dependency is loaded only when this exporter is instantiated.
    """

    def __init__(
        self,
        *,
        namespace: str = "asterinis",
    ) -> None:
        try:
            from prometheus_client import Counter, Histogram
        except ImportError as exc:
            raise ImportError(
                "Prometheus support requires the optional dependency. "
                'Install it with: pip install "asterinis[prometheus]"'
            ) from exc

        self.metrics = PrometheusMetrics(
            requests_total=Counter(
                f"{namespace}_requests_total",
                "Total number of Asterinis requests.",
                ["route"],
            ),
            request_duration_seconds=Histogram(
                f"{namespace}_request_duration_seconds",
                "Asterinis request duration in seconds.",
                ["route"],
            ),
            provider_errors_total=Counter(
                f"{namespace}_provider_errors_total",
                "Total number of provider errors.",
                ["provider"],
            ),
            agent_runs_total=Counter(
                f"{namespace}_agent_runs_total",
                "Total number of agent executions.",
                ["agent"],
            ),
            retrieval_duration_seconds=Histogram(
                f"{namespace}_retrieval_duration_seconds",
                "Retrieval duration in seconds.",
                ["retriever"],
            ),
        )

    def record_request(
        self,
        *,
        route: str,
        duration_seconds: float,
    ) -> None:
        self.metrics.requests_total.labels(
            route=route
        ).inc()

        self.metrics.request_duration_seconds.labels(
            route=route
        ).observe(duration_seconds)

    def record_provider_error(
        self,
        provider: str,
    ) -> None:
        self.metrics.provider_errors_total.labels(
            provider=provider
        ).inc()

    def record_agent_run(
        self,
        agent: str,
    ) -> None:
        self.metrics.agent_runs_total.labels(
            agent=agent
        ).inc()

    def record_retrieval(
        self,
        *,
        retriever: str,
        duration_seconds: float,
    ) -> None:
        self.metrics.retrieval_duration_seconds.labels(
            retriever=retriever
        ).observe(duration_seconds)