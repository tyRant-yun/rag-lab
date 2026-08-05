from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
    index_text: str | None = None,
    heading_path: list[str] | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=index_text or f"正文 {chunk_id}",
        index_text=index_text or f"索引正文 {chunk_id}",
        heading_path=heading_path or ["第一章"],
        page_start=ordinal,
        page_end=ordinal,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def write_chunks(
    path: Path,
    chunks: Sequence[KnowledgeChunk],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                chunk.to_dict(),
                ensure_ascii=False,
            )
            for chunk in chunks
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        dimensions: int = 2,
    ) -> None:
        self._dimensions = dimensions
        self.query_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def embedding_version(self) -> str:
        return (
            "fake:fake-model:"
            f"{self.dimensions}:v1"
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        del texts
        raise AssertionError(
            "search CLIs must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        self.query_calls.append(text)

        return EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=self.dimensions,
        )


class FakeVectorStore:
    def __init__(
        self,
        *,
        collection_name: str,
        dimensions: int = 2,
        candidate_count: int = 0,
        matches: Sequence[VectorMatch] = (),
    ) -> None:
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._candidate_count = candidate_count
        self._matches = list(matches)
        self.count_calls: list[SearchFilters | None] = []
        self.search_calls: list[
            tuple[
                EmbeddingVector,
                int,
                SearchFilters | None,
            ]
        ] = []

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def ensure_collection(self) -> None:
        raise AssertionError(
            "search CLIs must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "search CLIs must not index records"
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        self.count_calls.append(filters)
        return self._candidate_count

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        self.search_calls.append(
            (vector, top_k, filters)
        )
        return list(self._matches)
