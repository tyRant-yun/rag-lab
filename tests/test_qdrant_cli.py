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
from rag_lab.vector_store.cli import (
    index_chunks,
    main,
)


def make_chunk(
    chunk_id: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=f"正文 {chunk_id}",
        index_text=f"索引正文 {chunk_id}",
        heading_path=["第一章"],
        page_start=ordinal,
        page_end=ordinal,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_chunks() -> tuple[KnowledgeChunk, ...]:
    return (
        make_chunk("chunk-1", 1),
        make_chunk("chunk-2", 2),
        make_chunk("chunk-3", 3),
    )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def embedding_version(self) -> str:
        return "fake:fake-model:2:v1"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        active_texts = list(texts)
        self.calls.append(active_texts)

        return EmbeddingBatch(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            vectors=[
                EmbeddingVector(
                    values=[0.6, 0.8],
                    dimensions=2,
                )
                for _ in active_texts
            ],
            input_count=len(active_texts),
            elapsed_ms=1.0,
            embedding_version=(
                self.embedding_version
            ),
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=2,
        )


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    @property
    def collection_name(self) -> str:
        return "test-collection"

    @property
    def dimensions(self) -> int:
        return 2

    def ensure_collection(self) -> None:
        return None

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        self.records = list(records)

        return VectorWriteReport(
            collection_name=self.collection_name,
            dimensions=self.dimensions,
            input_count=len(self.records),
            upserted_count=len(self.records),
            elapsed_ms=1.0,
            embedding_version=(
                self.records[0].embedding_version
            ),
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        del filters
        return len(self.records)

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector
        del top_k
        del filters
        return []


def test_index_chunks_batches_and_upserts():
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    report = index_chunks(
        chunks=make_chunks(),
        provider=provider,
        store=store,
        batch_size=2,
    )

    assert provider.calls == [
        [
            "索引正文 chunk-1",
            "索引正文 chunk-2",
        ],
        [
            "索引正文 chunk-3",
        ],
    ]
    assert len(store.records) == 3
    assert report.upserted_count == 3


def test_cli_writes_json_report(
    tmp_path: Path,
    capsys,
):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(
            json.dumps(
                chunk.to_dict(),
                ensure_ascii=False,
            )
            for chunk in make_chunks()
        )
        + "\n",
        encoding="utf-8",
    )

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--collection",
            "test-collection",
            "--dimensions",
            "2",
            "--batch-size",
            "2",
            "--json",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["collection_name"] == (
        "test-collection"
    )
    assert payload["input_count"] == 3
    assert payload["upserted_count"] == 3
