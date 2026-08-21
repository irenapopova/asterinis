from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .documents import RetrievalResult


@dataclass(slots=True)
class Citation:
    id: str
    document_id: str
    text: str
    source: str | None = None
    title: str | None = None
    url: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.document_id = self.document_id.strip()
        self.text = self.text.strip()

        if not self.id:
            raise ValueError(
                "Citation id cannot be empty."
            )

        if not self.document_id:
            raise ValueError(
                "Citation document_id cannot be empty."
            )

        if not self.text:
            raise ValueError(
                "Citation text cannot be empty."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CitationBundle:
    citations: list[Citation]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.citations)

    def get(
        self,
        citation_id: str,
    ) -> Citation | None:
        for citation in self.citations:
            if citation.id == citation_id:
                return citation

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "citations": [
                citation.to_dict()
                for citation in self.citations
            ],
            "metadata": dict(self.metadata),
        }


class CitationBuilder:
    """
    Builds citation objects from retrieval results.

    Source information is read from document metadata when available.
    By default, the builder looks for:

    - source
    - title
    - url

    Applications can use different metadata keys if needed.
    """

    def __init__(
        self,
        *,
        source_key: str = "source",
        title_key: str = "title",
        url_key: str = "url",
    ) -> None:
        self.source_key = source_key
        self.title_key = title_key
        self.url_key = url_key

    def build(
        self,
        results: Iterable[RetrievalResult],
        *,
        prefix: str = "source",
        metadata: dict[str, Any] | None = None,
    ) -> CitationBundle:
        prefix = prefix.strip()

        if not prefix:
            raise ValueError(
                "Citation prefix cannot be empty."
            )

        citations: list[Citation] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            document = result.document
            document_metadata = document.metadata

            citations.append(
                Citation(
                    id=f"{prefix}-{index}",
                    document_id=document.id,
                    text=document.text,
                    source=self._optional_string(
                        document_metadata.get(
                            self.source_key
                        )
                    ),
                    title=self._optional_string(
                        document_metadata.get(
                            self.title_key
                        )
                    ),
                    url=self._optional_string(
                        document_metadata.get(
                            self.url_key
                        )
                    ),
                    score=float(result.score),
                    metadata={
                        "retriever": result.retriever,
                        **dict(result.metadata),
                    },
                )
            )

        return CitationBundle(
            citations=citations,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        value = str(value).strip()

        return value or None