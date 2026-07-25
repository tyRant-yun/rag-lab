from __future__ import annotations

import json
from pathlib import Path

from rag_lab.chunking import (
    ChunkingReport,
    ChunkingResult,
)
from rag_lab.chunking.serialization import (
    write_chunking_outputs,
)
from rag_lab.contracts import KnowledgeChunk


def build_result() -> ChunkingResult:
    chunk = KnowledgeChunk(
        chunk_id="sha256:chunk",
        document_id="sha256:document",
        content="因特网是一个网络的网络。",
        index_text=(
            "第1章 计算机网络和因特网\n"
            "1.1 什么是因特网\n\n"
            "因特网是一个网络的网络。"
        ),
        heading_path=[
            "第1章 计算机网络和因特网",
            "1.1 什么是因特网",
        ],
        page_start=19,
        page_end=20,
        ordinal=1,
        block_ids=[
            "sha256:block-1",
            "sha256:block-2",
        ],
        source_path="D:/source.pdf",
        content_hash="sha256:content",
        normalization_version="1.0.0",
        chunking_version="1.0.0",
    )

    report = ChunkingReport(
        document_id="sha256:document",
        input_block_count=4,
        output_chunk_count=1,
        cross_page_join_count=1,
        long_block_split_count=0,
        oversized_atomic_block_count=0,
        overlapped_chunk_count=0,
        overlap_char_count=0,
    )

    return ChunkingResult(
        chunks=[chunk],
        report=report,
    )


def test_writes_all_chunking_artifacts(
    tmp_path: Path,
):
    write_chunking_outputs(
        result=build_result(),
        output_directory=tmp_path,
    )

    assert {
        path.name
        for path in tmp_path.iterdir()
    } == {
        "chunks.jsonl",
        "chunks.md",
        "chunking-report.json",
    }


def test_chunks_jsonl_is_utf8_and_parseable(
    tmp_path: Path,
):
    write_chunking_outputs(
        result=build_result(),
        output_directory=tmp_path,
    )

    path = tmp_path / "chunks.jsonl"
    content = path.read_text(
        encoding="utf-8"
    )

    lines = content.splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["chunk_id"] == (
        "sha256:chunk"
    )
    assert record["content"] == (
        "因特网是一个网络的网络。"
    )
    assert record["block_ids"] == [
        "sha256:block-1",
        "sha256:block-2",
    ]


def test_chunks_markdown_contains_review_metadata(
    tmp_path: Path,
):
    write_chunking_outputs(
        result=build_result(),
        output_directory=tmp_path,
    )

    content = (
        tmp_path / "chunks.md"
    ).read_text(encoding="utf-8")

    assert "## Chunk 1" in content
    assert "sha256:chunk" in content
    assert "第1章 计算机网络和因特网" in content
    assert "pages=19-20" in content
    assert "因特网是一个网络的网络。" in content


def test_chunking_report_is_parseable(
    tmp_path: Path,
):
    write_chunking_outputs(
        result=build_result(),
        output_directory=tmp_path,
    )

    report = json.loads(
        (
            tmp_path
            / "chunking-report.json"
        ).read_text(encoding="utf-8")
    )

    assert report == {
        "document_id": "sha256:document",
        "input_block_count": 4,
        "output_chunk_count": 1,
        "cross_page_join_count": 1,
        "long_block_split_count": 0,
        "oversized_atomic_block_count": 0,
        "overlapped_chunk_count": 0,
        "overlap_char_count": 0,
    }


def test_empty_chunks_create_empty_review_files(
    tmp_path: Path,
):
    result = ChunkingResult(
        chunks=[],
        report=ChunkingReport(
            document_id="sha256:document",
            input_block_count=2,
            output_chunk_count=0,
            cross_page_join_count=0,
            long_block_split_count=0,
            oversized_atomic_block_count=0,
            overlapped_chunk_count=0,
            overlap_char_count=0,
        ),
    )

    write_chunking_outputs(
        result=result,
        output_directory=tmp_path,
    )

    assert (
        tmp_path / "chunks.jsonl"
    ).read_text(encoding="utf-8") == ""

    assert (
        tmp_path / "chunks.md"
    ).read_text(encoding="utf-8") == ""


def test_repeated_write_is_deterministic(
    tmp_path: Path,
):
    result = build_result()

    write_chunking_outputs(
        result=result,
        output_directory=tmp_path,
    )

    first = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    write_chunking_outputs(
        result=result,
        output_directory=tmp_path,
    )

    second = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    assert first == second
