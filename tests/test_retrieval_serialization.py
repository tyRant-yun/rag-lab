from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lab.contracts import KnowledgeChunk
from rag_lab.retrieval import (
    read_knowledge_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        content=f"Chunk {ordinal} 正文",
        index_text=f"第一章 Chunk {ordinal} 正文",
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


def write_jsonl(
    path: Path,
    payloads: list[dict[str, object]],
) -> None:
    content = "\n".join(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
        for payload in payloads
    )

    if payloads:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def test_reads_chunks_in_file_order(
    tmp_path: Path,
):
    first = make_chunk(
        chunk_id="chunk-1",
        ordinal=1,
    )
    second = make_chunk(
        chunk_id="chunk-2",
        ordinal=2,
    )
    path = tmp_path / "chunks.jsonl"

    write_jsonl(
        path,
        [
            first.to_dict(),
            second.to_dict(),
        ],
    )

    chunks = read_knowledge_chunks_jsonl(
        path
    )

    assert chunks == (first, second)


def test_empty_file_returns_empty_tuple(
    tmp_path: Path,
):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    assert read_knowledge_chunks_jsonl(
        path
    ) == ()


def test_rejects_empty_jsonl_record(
    tmp_path: Path,
):
    first = make_chunk(
        chunk_id="chunk-1",
        ordinal=1,
    )
    path = tmp_path / "chunks.jsonl"

    path.write_text(
        json.dumps(
            first.to_dict(),
            ensure_ascii=False,
        )
        + "\n\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        ValueError,
        match=r"line 2: empty JSONL record",
    ):
        read_knowledge_chunks_jsonl(path)


def test_reports_invalid_json_line(
    tmp_path: Path,
):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"chunk_id":\n',
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        ValueError,
        match=r"line 1: invalid JSON",
    ):
        read_knowledge_chunks_jsonl(path)


def test_reports_invalid_chunk_line(
    tmp_path: Path,
):
    chunk = make_chunk(
        chunk_id="chunk-1",
        ordinal=1,
    )
    payload = chunk.to_dict()
    payload["chunk_id"] = ""

    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [payload])

    with pytest.raises(
        ValueError,
        match=r"line 1: invalid KnowledgeChunk",
    ):
        read_knowledge_chunks_jsonl(path)


def test_uses_strict_validation(
    tmp_path: Path,
):
    chunk = make_chunk(
        chunk_id="chunk-1",
        ordinal=1,
    )
    payload = chunk.to_dict()
    payload["page_start"] = "1"

    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [payload])

    with pytest.raises(
        ValueError,
        match=r"line 1: invalid KnowledgeChunk",
    ):
        read_knowledge_chunks_jsonl(path)
