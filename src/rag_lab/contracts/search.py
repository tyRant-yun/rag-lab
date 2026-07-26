from __future__ import annotations

import math
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from rag_lab.contracts.chunks import (
    KnowledgeChunk,
)


class SearchFilters(BaseModel):
    """Storage-neutral constraints applied to retrieval."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    document_ids: list[str] | None = None
    heading_prefix: list[str] | None = None

    page_start: int | None = None
    page_end: int | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.document_ids is not None:
            if not self.document_ids:
                raise ValueError(
                    "document_ids cannot be empty"
                )

            if any(
                not document_id.strip()
                for document_id
                in self.document_ids
            ):
                raise ValueError(
                    "document_ids entries cannot "
                    "be empty"
                )

            if len(set(self.document_ids)) != len(
                self.document_ids
            ):
                raise ValueError(
                    "document_ids cannot contain "
                    "duplicates"
                )

        if self.heading_prefix is not None:
            if not self.heading_prefix:
                raise ValueError(
                    "heading_prefix cannot be empty"
                )

            if any(
                not heading.strip()
                for heading
                in self.heading_prefix
            ):
                raise ValueError(
                    "heading_prefix entries cannot "
                    "be empty"
                )

        if (
            self.page_start is not None
            and self.page_start < 1
        ):
            raise ValueError(
                "page_start must be at least 1"
            )

        if (
            self.page_end is not None
            and self.page_end < 1
        ):
            raise ValueError(
                "page_end must be at least 1"
            )

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must not precede "
                "page_start"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class SearchHit(BaseModel):
    """One ranked source-grounded retrieval result."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    chunk: KnowledgeChunk
    score: float
    rank: int
    retriever: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not math.isfinite(self.score):
            raise ValueError(
                "score must be finite"
            )

        if self.rank < 1:
            raise ValueError(
                "rank must be at least 1"
            )

        if not self.retriever.strip():
            raise ValueError(
                "retriever cannot be empty"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

class SearchResult(BaseModel):
    """Complete output from one retrieval operation."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    query: str
    hits: list[SearchHit]

    candidate_count: int
    elapsed_ms: float

    retriever: str
    index_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if self.candidate_count < 0:
            raise ValueError(
                "candidate_count cannot be negative"
            )

        if self.candidate_count < len(self.hits):
            raise ValueError(
                "candidate_count cannot be less "
                "than hit count"
            )

        if (
            not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0
        ):
            raise ValueError(
                "elapsed_ms must be finite and "
                "non-negative"
            )

        if not self.retriever.strip():
            raise ValueError(
                "retriever cannot be empty"
            )

        if not self.index_version.strip():
            raise ValueError(
                "index_version cannot be empty"
            )

        chunk_ids = [
            hit.chunk.chunk_id
            for hit in self.hits
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                "hits cannot contain duplicate "
                "chunk IDs"
            )

        expected_ranks = list(
            range(1, len(self.hits) + 1)
        )
        actual_ranks = [
            hit.rank
            for hit in self.hits
        ]

        if actual_ranks != expected_ranks:
            raise ValueError(
                "hit ranks must be contiguous "
                "and ordered from 1"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
