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
from rag_lab.contracts.embeddings import (
    EmbeddingVector,
)


class VectorRecord(BaseModel):
    """One Chunk and its validated embedding vector."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    chunk: KnowledgeChunk
    vector: EmbeddingVector
    embedding_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.embedding_version.strip():
            raise ValueError(
                "embedding_version cannot be empty"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class VectorMatch(BaseModel):
    """One storage-neutral vector similarity match."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    chunk: KnowledgeChunk
    score: float

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not math.isfinite(self.score):
            raise ValueError(
                "score must be finite"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class VectorWriteReport(BaseModel):
    """Summary of one completed vector upsert operation."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    collection_name: str
    dimensions: int
    input_count: int
    upserted_count: int
    elapsed_ms: float
    embedding_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty"
            )

        if not self.embedding_version.strip():
            raise ValueError(
                "embedding_version cannot be empty"
            )

        if self.dimensions < 1:
            raise ValueError(
                "dimensions must be at least 1"
            )

        if self.input_count < 1:
            raise ValueError(
                "input_count must be at least 1"
            )

        if self.upserted_count != self.input_count:
            raise ValueError(
                "upserted_count must equal input_count"
            )

        if (
            not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0
        ):
            raise ValueError(
                "elapsed_ms must be finite and "
                "non-negative"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
