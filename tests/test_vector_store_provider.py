from __future__ import annotations

from collections.abc import Sequence

from rag_lab.contracts import (
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.vector_store import VectorStore


def make_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id="chunk-network-protocol",
        document_id="computer-networking",
        content="网络协议规定报文的格式和顺序。",
        index_text=(
            "第一章 网络协议 "
            "网络协议规定报文的格式和顺序。"
        ),
        heading_path=[
            "第一章",
            "网络协议",
        ],
        page_start=19,
        page_end=19,
        ordinal=1,
        block_ids=["block-001"],
        source_path="computer-networking.pdf",
        content_hash="content-hash-001",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_record() -> VectorRecord:
    return VectorRecord(
        chunk=make_chunk(),
        vector=EmbeddingVector(
            values=[0.1, 0.2, 0.3],
            dimensions=3,
        ),
        embedding_version="fake:model:3:v1",
    )


class FakeVectorStore:
    """In-memory test double for VectorStore."""

    def __init__(self) -> None:
        self._collection_name = "test-collection"
        self._dimensions = 3
        self._records: list[VectorRecord] = []
        self.collection_ready = False

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def ensure_collection(self) -> None:
        self.collection_ready = True

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        self._records = list(records)

        return VectorWriteReport(
            collection_name=self.collection_name,
            dimensions=self.dimensions,
            input_count=len(records),
            upserted_count=len(records),
            elapsed_ms=0.0,
            embedding_version=(
                records[0].embedding_version
            ),
        )

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector
        del filters

        return [
            VectorMatch(
                chunk=record.chunk,
                score=1.0,
            )
            for record in self._records[:top_k]
        ]


def test_fake_store_matches_vector_store_protocol():
    store = FakeVectorStore()

    assert isinstance(store, VectorStore)


def test_incomplete_object_does_not_match_protocol():
    assert not isinstance(object(), VectorStore)


def test_vector_store_exposes_identity():
    store: VectorStore = FakeVectorStore()

    assert store.collection_name == (
        "test-collection"
    )
    assert store.dimensions == 3


def test_vector_store_ensures_collection():
    store = FakeVectorStore()

    store.ensure_collection()

    assert store.collection_ready is True


def test_vector_store_upserts_records():
    store: VectorStore = FakeVectorStore()
    record = make_record()

    report = store.upsert([record])

    assert report.input_count == 1
    assert report.upserted_count == 1


def test_vector_store_searches_records():
    store: VectorStore = FakeVectorStore()
    record = make_record()

    store.upsert([record])

    matches = store.search(
        EmbeddingVector(
            values=[0.1, 0.2, 0.3],
            dimensions=3,
        ),
        top_k=1,
    )

    assert len(matches) == 1
    assert matches[0].chunk.chunk_id == (
        "chunk-network-protocol"
    )
    assert matches[0].score == 1.0


def test_vector_store_is_publicly_exported():
    import rag_lab.vector_store as vector_store

    assert "VectorStore" in vector_store.__all__
