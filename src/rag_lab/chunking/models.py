from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.contracts import KnowledgeChunk


class ChunkingConfig(BaseModel):
    """Configuration for deterministic chunk generation."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    # 包含标题路径后的 index_text 最大字符数。
    max_chars: int = 1200
    chunking_version: str = "1.0.0"

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.max_chars < 100:
            raise ValueError(
                "max_chars must be at least 100"
            )

        if not self.chunking_version.strip():
            raise ValueError(
                "chunking_version cannot be empty"
            )

        return self


class ChunkingReport(BaseModel):
    """Statistics produced by one chunking operation."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    document_id: str
    input_block_count: int = Field(ge=0)
    output_chunk_count: int = Field(ge=0)
    cross_page_join_count: int = Field(ge=0)
    long_block_split_count: int = Field(ge=0)
    oversized_atomic_block_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if not self.document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        return self


class ChunkingResult(BaseModel):
    """Chunks and diagnostics returned together."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    chunks: list[KnowledgeChunk]
    report: ChunkingReport
