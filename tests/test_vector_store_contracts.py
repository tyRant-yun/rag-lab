from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from rag_lab.contracts import (
    EmbeddingVector,
    KnowledgeChunk,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)


def make_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id="chunk-network-protocol",
        document_id="computer-networking",
        content="网络协议定义通信实体之间交换报文的格式和顺序。",
        index_text=(
            "第一章 计算机网络和因特网 "
            "网络协议定义通信实体之间交换报文的格式和顺序。"
        ),
        heading_path=[
            "第一章",
            "计算机网络和因特网",
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


def make_vector() -> EmbeddingVector:
    return EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        dimensions=3,
    )


def test_vector_record_accepts_valid_data():
    record = VectorRecord(
        chunk=make_chunk(),
        vector=make_vector(),
        embedding_version="ollama:qwen3-embedding:3:v1",
    )

    assert record.chunk.chunk_id == (
        "chunk-network-protocol"
    )
    assert record.vector.dimensions == 3
    assert record.embedding_version == (
        "ollama:qwen3-embedding:3:v1"
    )


def test_vector_record_rejects_empty_embedding_version():
    with pytest.raises(
        ValidationError,
        match="embedding_version cannot be empty",
    ):
        VectorRecord(
            chunk=make_chunk(),
            vector=make_vector(),
            embedding_version=" ",
        )


def test_vector_match_accepts_finite_score():
    match = VectorMatch(
        chunk=make_chunk(),
        score=0.875,
    )

    assert match.score == 0.875


@pytest.mark.parametrize(
    "score",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_vector_match_rejects_non_finite_score(
    score: float,
):
    with pytest.raises(
        ValidationError,
        match="score must be finite",
    ):
        VectorMatch(
            chunk=make_chunk(),
            score=score,
        )


def test_vector_write_report_accepts_completed_write():
    report = VectorWriteReport(
        collection_name="computer-networking",
        dimensions=1024,
        input_count=8,
        upserted_count=8,
        elapsed_ms=25.5,
        embedding_version=(
            "ollama:qwen3-embedding:0.6b:1024:v1"
        ),
    )

    assert report.input_count == 8
    assert report.upserted_count == 8


def test_vector_write_report_rejects_count_mismatch():
    with pytest.raises(
        ValidationError,
        match=(
            "upserted_count must equal input_count"
        ),
    ):
        VectorWriteReport(
            collection_name="computer-networking",
            dimensions=1024,
            input_count=8,
            upserted_count=7,
            elapsed_ms=25.5,
            embedding_version=(
                "ollama:qwen3-embedding:0.6b:1024:v1"
            ),
        )


def test_vector_store_contracts_are_publicly_exported():
    import rag_lab.contracts as contracts

    assert "VectorRecord" in contracts.__all__
    assert "VectorMatch" in contracts.__all__
    assert "VectorWriteReport" in contracts.__all__
