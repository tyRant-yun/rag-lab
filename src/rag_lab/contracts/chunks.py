from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class KnowledgeChunk(BaseModel):
    """A stable unit passed from Chunker to retrieval/indexing."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    chunk_id: str
    document_id: str

    # 用于回答时展示和引用的正文。
    content: str

    # 用于 BM25 和 Embedding 的完整索引文本，
    # 通常由标题路径和正文组成。
    index_text: str

    heading_path: list[str]

    page_start: int
    page_end: int

    # 当前文档内的 Chunk 顺序，从 1 开始。
    ordinal: int

    # 组成此 Chunk 的 NormalizedBlock ID。
    block_ids: list[str]

    source_path: str
    content_hash: str
    normalization_version: str
    chunking_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        required_strings = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "index_text": self.index_text,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "normalization_version": self.normalization_version,
            "chunking_version": self.chunking_version,
        }

        for field_name, value in required_strings.items():
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")

        if not self.heading_path:
            raise ValueError("heading_path cannot be empty")

        if any(
            not heading.strip()
            for heading in self.heading_path
        ):
            raise ValueError(
                "heading_path entries cannot be empty"
            )

        if not self.block_ids:
            raise ValueError("block_ids cannot be empty")

        if any(
            not block_id.strip()
            for block_id in self.block_ids
        ):
            raise ValueError(
                "block_ids entries cannot be empty"
            )

        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError(
                "block_ids cannot contain duplicates"
            )

        if self.page_start < 1:
            raise ValueError(
                "page_start must be at least 1"
            )

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end must not precede page_start"
            )

        if self.ordinal < 1:
            raise ValueError(
                "ordinal must be at least 1"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
