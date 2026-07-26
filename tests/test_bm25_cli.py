from __future__ import annotations

import json
from pathlib import Path

from rag_lab.contracts import KnowledgeChunk
from rag_lab.retrieval.bm25.cli import main


def make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    index_text: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=index_text,
        index_text=index_text,
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


def write_chunks(
    path: Path,
    chunks: tuple[KnowledgeChunk, ...],
) -> None:
    content = "\n".join(
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
        )
        for chunk in chunks
    )

    if chunks:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def make_chunks_file(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "chunks.jsonl"

    write_chunks(
        path,
        (
            make_chunk(
                chunk_id="chunk-tcp",
                document_id="document-a",
                index_text=(
                    "TCP 使用拥塞控制保证网络性能"
                ),
                ordinal=1,
            ),
            make_chunk(
                chunk_id="chunk-http",
                document_id="document-a",
                index_text=(
                    "HTTP 支持 Web 应用通信"
                ),
                ordinal=2,
            ),
            make_chunk(
                chunk_id="chunk-dns",
                document_id="document-b",
                index_text=(
                    "DNS 将域名转换为 IP 地址"
                ),
                ordinal=3,
            ),
        ),
    )

    return path


def test_writes_human_readable_results(
    tmp_path: Path,
    capsys,
):
    path = make_chunks_file(tmp_path)

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--query",
            "TCP 拥塞控制",
            "--user-word",
            "拥塞控制",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Query: TCP 拥塞控制" in captured.out
    assert "chunk_id=chunk-tcp" in captured.out
    assert "pages=1-1" in captured.out


def test_writes_json_result(
    tmp_path: Path,
    capsys,
):
    path = make_chunks_file(tmp_path)

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--query",
            "DNS",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["retriever"] == "bm25"
    assert payload["hits"][0]["chunk"]["chunk_id"] == (
        "chunk-dns"
    )


def test_applies_document_filter(
    tmp_path: Path,
    capsys,
):
    path = make_chunks_file(tmp_path)

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--query",
            "DNS",
            "--document-id",
            "document-b",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "chunk_id=chunk-dns" in captured.out
    assert "Candidates: 1" in captured.out


def test_reports_no_matching_chunks(
    tmp_path: Path,
    capsys,
):
    path = make_chunks_file(tmp_path)

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--query",
            "量子纠缠",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No matching chunks." in captured.out


def test_reports_missing_file(
    tmp_path: Path,
    capsys,
):
    missing = tmp_path / "missing.jsonl"

    exit_code = main(
        [
            "--chunks",
            str(missing),
            "--query",
            "TCP",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_reports_empty_chunk_file(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--query",
            "TCP",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "chunks cannot be empty" in captured.err
