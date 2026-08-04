from __future__ import annotations

from uuid import UUID

from qdrant_client import models

from rag_lab.contracts import (
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorRecord,
)
from rag_lab.vector_store.payload import (
    chunk_point_id,
    filters_to_qdrant_filter,
    heading_prefix_key,
    heading_prefix_keys,
    payload_to_chunk,
    record_to_payload,
)


def make_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id="chunk-http",
        document_id="computer-networking",
        content="HTTP 是应用层协议。",
        index_text=(
            "第二章 应用层 HTTP "
            "HTTP 是应用层协议。"
        ),
        heading_path=[
            "第二章",
            "应用层",
            "HTTP",
        ],
        page_start=20,
        page_end=22,
        ordinal=2,
        block_ids=["block-020"],
        source_path="computer-networking.pdf",
        content_hash="hash-http",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_record() -> VectorRecord:
    return VectorRecord(
        chunk=make_chunk(),
        vector=EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=2,
        ),
        embedding_version="fake:model:2:v1",
    )


def condition_for(
    qdrant_filter: models.Filter,
    key: str,
) -> models.FieldCondition:
    assert qdrant_filter.must is not None

    for condition in qdrant_filter.must:
        if (
            isinstance(
                condition,
                models.FieldCondition,
            )
            and condition.key == key
        ):
            return condition

    raise AssertionError(
        f"missing condition for {key}"
    )


def test_chunk_point_id_is_stable_uuid():
    first = chunk_point_id("chunk-http")
    second = chunk_point_id("chunk-http")

    assert first == second
    assert str(UUID(first)) == first


def test_different_chunks_have_different_point_ids():
    assert (
        chunk_point_id("chunk-http")
        != chunk_point_id("chunk-dns")
    )


def test_builds_ordered_heading_prefixes():
    prefixes = heading_prefix_keys(
        [
            "第二章",
            "应用层",
            "HTTP",
        ]
    )

    assert prefixes == [
        '["第二章"]',
        '["第二章","应用层"]',
        '["第二章","应用层","HTTP"]',
    ]


def test_payload_round_trip_restores_chunk():
    record = make_record()

    payload = record_to_payload(record)
    restored = payload_to_chunk(payload)

    assert restored == record.chunk
    assert payload["embedding_version"] == (
        "fake:model:2:v1"
    )
    assert payload["heading_prefixes"] == [
        '["第二章"]',
        '["第二章","应用层"]',
        '["第二章","应用层","HTTP"]',
    ]


def test_empty_filters_return_none():
    assert filters_to_qdrant_filter(
        SearchFilters()
    ) is None


def test_converts_document_filter():
    result = filters_to_qdrant_filter(
        SearchFilters(
            document_ids=[
                "document-a",
                "document-b",
            ],
        )
    )

    assert result is not None

    condition = condition_for(
        result,
        "document_id",
    )

    assert condition.match == models.MatchAny(
        any=[
            "document-a",
            "document-b",
        ]
    )


def test_converts_heading_prefix_filter():
    result = filters_to_qdrant_filter(
        SearchFilters(
            heading_prefix=[
                "第二章",
                "应用层",
            ],
        )
    )

    assert result is not None

    condition = condition_for(
        result,
        "heading_prefixes",
    )

    assert condition.match == models.MatchValue(
        value=heading_prefix_key(
            [
                "第二章",
                "应用层",
            ]
        )
    )


def test_converts_page_overlap_filter():
    result = filters_to_qdrant_filter(
        SearchFilters(
            page_start=21,
            page_end=25,
        )
    )

    assert result is not None

    page_end_condition = condition_for(
        result,
        "page_end",
    )
    page_start_condition = condition_for(
        result,
        "page_start",
    )

    assert page_end_condition.range is not None
    assert page_end_condition.range.gte == 21.0

    assert page_start_condition.range is not None
    assert page_start_condition.range.lte == 25.0
