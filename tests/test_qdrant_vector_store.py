from __future__ import annotations

import pytest
from qdrant_client import (
    QdrantClient,
    models,
)

from rag_lab.contracts import (
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorRecord,
)
from rag_lab.vector_store import (
    QdrantCollectionConfigurationError,
    QdrantVectorStore,
    QdrantVectorStoreError,
)


def make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    heading_path: list[str],
    page_start: int,
    page_end: int | None = None,
) -> KnowledgeChunk:
    resolved_page_end = (
        page_start
        if page_end is None
        else page_end
    )

    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"正文 {chunk_id}",
        index_text=f"索引正文 {chunk_id}",
        heading_path=heading_path,
        page_start=page_start,
        page_end=resolved_page_end,
        ordinal=page_start,
        block_ids=[f"block-{chunk_id}"],
        source_path="book.pdf",
        content_hash=f"hash-{chunk_id}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_record(
    *,
    chunk_id: str,
    document_id: str,
    heading_path: list[str],
    page_start: int,
    page_end: int | None = None,
    values: list[float],
) -> VectorRecord:
    return VectorRecord(
        chunk=make_chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
        ),
        vector=EmbeddingVector(
            values=values,
            dimensions=3,
        ),
        embedding_version="fake:model:3:v1",
    )


def make_store(
    client: QdrantClient,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name="test-chunks",
        dimensions=3,
    )


def test_ensure_collection_is_idempotent():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.ensure_collection()
    store.ensure_collection()

    assert client.collection_exists(
        "test-chunks"
    )


def test_rejects_existing_dimension_mismatch():
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="test-chunks",
        vectors_config=models.VectorParams(
            size=4,
            distance=models.Distance.COSINE,
        ),
    )

    store = make_store(client)

    with pytest.raises(
        QdrantCollectionConfigurationError,
        match="dimensions mismatch",
    ):
        store.ensure_collection()


def test_upsert_is_idempotent_by_chunk_id():
    client = QdrantClient(":memory:")
    store = make_store(client)

    record = make_record(
        chunk_id="chunk-http",
        document_id="document-a",
        heading_path=["第二章", "应用层"],
        page_start=20,
        values=[1.0, 0.0, 0.0],
    )

    first_report = store.upsert([record])
    second_report = store.upsert([record])

    count = client.count(
        collection_name="test-chunks",
        exact=True,
    )

    assert first_report.upserted_count == 1
    assert second_report.upserted_count == 1
    assert count.count == 1


def test_count_returns_zero_for_empty_collection():
    client = QdrantClient(":memory:")
    store = make_store(client)

    assert store.count() == 0


def test_count_returns_all_points():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.upsert(
        [
            make_record(
                chunk_id="chunk-http",
                document_id="document-a",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=20,
                values=[1.0, 0.0, 0.0],
            ),
            make_record(
                chunk_id="chunk-dns",
                document_id="document-b",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=24,
                values=[0.8, 0.2, 0.0],
            ),
        ]
    )

    assert store.count() == 2


def test_count_applies_existing_filter_semantics():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.upsert(
        [
            make_record(
                chunk_id="chunk-http",
                document_id="document-a",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=20,
                values=[1.0, 0.0, 0.0],
            ),
            make_record(
                chunk_id="chunk-dns",
                document_id="document-b",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=24,
                values=[0.8, 0.2, 0.0],
            ),
        ]
    )

    count = store.count(
        filters=SearchFilters(
            document_ids=["document-b"],
            heading_prefix=[
                "第二章",
                "应用层",
            ],
            page_start=21,
            page_end=25,
        )
    )

    assert count == 1


def test_count_uses_page_interval_overlap():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.upsert(
        [
            make_record(
                chunk_id="chunk-a",
                document_id="document-a",
                heading_path=["第一章"],
                page_start=19,
                page_end=22,
                values=[1.0, 0.0, 0.0],
            ),
            make_record(
                chunk_id="chunk-b",
                document_id="document-a",
                heading_path=["第一章"],
                page_start=24,
                page_end=30,
                values=[0.8, 0.2, 0.0],
            ),
        ]
    )

    count = store.count(
        filters=SearchFilters(
            page_start=21,
            page_end=25,
        )
    )

    assert count == 2


def test_count_wraps_client_error(
    monkeypatch: pytest.MonkeyPatch,
):
    client = QdrantClient(":memory:")
    store = make_store(client)
    store.ensure_collection()

    original_error = RuntimeError(
        "count service unavailable"
    )

    def raise_count_error(
        *args: object,
        **kwargs: object,
    ) -> None:
        del args
        del kwargs
        raise original_error

    monkeypatch.setattr(
        client,
        "count",
        raise_count_error,
    )

    with pytest.raises(
        QdrantVectorStoreError,
        match="failed to count Qdrant points",
    ) as error_info:
        store.count()

    assert error_info.value.__cause__ is original_error


def test_search_restores_ranked_chunks():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.upsert(
        [
            make_record(
                chunk_id="chunk-http",
                document_id="document-a",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=20,
                values=[1.0, 0.0, 0.0],
            ),
            make_record(
                chunk_id="chunk-dns",
                document_id="document-b",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=24,
                values=[0.8, 0.2, 0.0],
            ),
        ]
    )

    matches = store.search(
        EmbeddingVector(
            values=[1.0, 0.0, 0.0],
            dimensions=3,
        ),
        top_k=2,
    )

    assert [
        match.chunk.chunk_id
        for match in matches
    ] == [
        "chunk-http",
        "chunk-dns",
    ]


def test_search_applies_existing_filter_semantics():
    client = QdrantClient(":memory:")
    store = make_store(client)

    store.upsert(
        [
            make_record(
                chunk_id="chunk-http",
                document_id="document-a",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=20,
                values=[1.0, 0.0, 0.0],
            ),
            make_record(
                chunk_id="chunk-dns",
                document_id="document-b",
                heading_path=[
                    "第二章",
                    "应用层",
                ],
                page_start=24,
                values=[0.8, 0.2, 0.0],
            ),
        ]
    )

    matches = store.search(
        EmbeddingVector(
            values=[1.0, 0.0, 0.0],
            dimensions=3,
        ),
        filters=SearchFilters(
            document_ids=["document-b"],
            heading_prefix=[
                "第二章",
                "应用层",
            ],
            page_start=21,
            page_end=25,
        ),
    )

    assert len(matches) == 1
    assert matches[0].chunk.chunk_id == (
        "chunk-dns"
    )


def test_search_rejects_dimension_mismatch():
    client = QdrantClient(":memory:")
    store = make_store(client)

    with pytest.raises(
        ValueError,
        match="query vector dimensions",
    ):
        store.search(
            EmbeddingVector(
                values=[0.6, 0.8],
                dimensions=2,
            )
        )
