from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .documents import RetrievalResult


MetadataPredicate = Callable[[dict[str, Any]], bool]


@dataclass(slots=True)
class MetadataFilter:
    """
    Represents a single metadata filtering rule.

    A rule can either use an exact field/value match or a custom predicate.
    """

    field: str | None = None
    value: Any = None
    predicate: MetadataPredicate | None = None

    def __post_init__(self) -> None:
        if self.field is None and self.predicate is None:
            raise ValueError(
                "MetadataFilter requires either a field or a predicate."
            )

        if self.field is not None:
            self.field = self.field.strip()

            if not self.field:
                raise ValueError(
                    "Metadata field cannot be empty."
                )

        if (
            self.predicate is not None
            and not callable(self.predicate)
        ):
            raise TypeError(
                "predicate must be callable."
            )

    def matches(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        if self.predicate is not None:
            return bool(
                self.predicate(metadata)
            )

        return metadata.get(self.field) == self.value


@dataclass(slots=True)
class MetadataFilterResult:
    accepted: list[RetrievalResult]
    rejected: list[RetrievalResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def total_count(self) -> int:
        return (
            self.accepted_count
            + self.rejected_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "total_count": self.total_count,
            "accepted": [
                result.to_dict()
                for result in self.accepted
            ],
            "rejected": [
                result.to_dict()
                for result in self.rejected
            ],
            "metadata": dict(self.metadata),
        }


class MetadataFilterSet:
    """
    Applies one or more metadata filters to retrieval results.

    Filters can be combined using either:

    - "all": every filter must match
    - "any": at least one filter must match
    """

    def __init__(
        self,
        filters: list[MetadataFilter],
        *,
        mode: str = "all",
    ) -> None:
        if not filters:
            raise ValueError(
                "At least one metadata filter is required."
            )

        if mode not in {"all", "any"}:
            raise ValueError(
                "mode must be either 'all' or 'any'."
            )

        self.filters = list(filters)
        self.mode = mode

    def apply(
        self,
        results: list[RetrievalResult],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MetadataFilterResult:
        accepted: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []

        for result in results:
            document_metadata = result.document.metadata

            matches = [
                filter_rule.matches(
                    document_metadata
                )
                for filter_rule in self.filters
            ]

            keep = (
                all(matches)
                if self.mode == "all"
                else any(matches)
            )

            if keep:
                accepted.append(result)
            else:
                rejected.append(result)

        return MetadataFilterResult(
            accepted=accepted,
            rejected=rejected,
            metadata={
                **(metadata or {}),
                "mode": self.mode,
                "filter_count": len(self.filters),
            },
        )