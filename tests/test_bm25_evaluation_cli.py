from __future__ import annotations

import json
from pathlib import Path

from rag_lab.contracts import KnowledgeChunk
from rag_lab.evaluation import (
    RetrievalEvaluationCase,
)
from rag_lab.evaluation.bm25_cli import main


def make_chunk(
    *,
    chunk_id: str,
    index_text: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-1",
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


def make_input_files(
    tmp_path: Path,
) -> tuple[Path, Path]:
    chunks_path = tmp_path / "chunks.jsonl"
    cases_path = tmp_path / "cases.jsonl"

    chunks = [
        make_chunk(
            chunk_id="chunk-tcp",
            index_text=(
                "TCP 使用拥塞控制保证网络性能"
            ),
            ordinal=1,
        ),
        make_chunk(
            chunk_id="chunk-http",
            index_text=(
                "HTTP 支持 Web 应用通信"
            ),
            ordinal=2,
        ),
        make_chunk(
            chunk_id="chunk-dns",
            index_text=(
                "DNS 将域名转换为 IP 地址"
            ),
            ordinal=3,
        ),
    ]

    cases = [
        RetrievalEvaluationCase(
            case_id="tcp-case",
            query="TCP",
            relevant_chunk_ids=[
                "chunk-tcp",
            ],
        ),
        RetrievalEvaluationCase(
            case_id="dns-case",
            query="DNS",
            relevant_chunk_ids=[
                "chunk-dns",
            ],
        ),
    ]

    write_jsonl(
        chunks_path,
        [
            chunk.to_dict()
            for chunk in chunks
        ],
    )
    write_jsonl(
        cases_path,
        [
            case.to_dict()
            for case in cases
        ],
    )

    return chunks_path, cases_path


def test_writes_human_readable_report(
    tmp_path: Path,
    capsys,
):
    chunks_path, cases_path = (
        make_input_files(tmp_path)
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dataset: test-dataset" in captured.out
    assert "Retriever: bm25" in captured.out
    assert "Top K: 2" in captured.out
    assert "Cases: 2" in captured.out
    assert "Hit@2: 1.000000" in captured.out
    assert "[tcp-case]" in captured.out
    assert "[dns-case]" in captured.out


def test_writes_json_report(
    tmp_path: Path,
    capsys,
):
    chunks_path, cases_path = (
        make_input_files(tmp_path)
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
            "--top-k",
            "2",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["dataset_id"] == (
        "test-dataset"
    )
    assert payload["retriever"] == "bm25"
    assert payload["top_k"] == 2
    assert payload["case_count"] == 2
    assert payload["hit_rate_at_k"] == 1.0
    assert len(payload["case_results"]) == 2


def test_rejects_unknown_relevant_chunk_id(
    tmp_path: Path,
    capsys,
):
    chunks_path, cases_path = (
        make_input_files(tmp_path)
    )

    unknown_case = RetrievalEvaluationCase(
        case_id="unknown-case",
        query="TCP",
        relevant_chunk_ids=[
            "chunk-missing",
        ],
    )
    write_jsonl(
        cases_path,
        [unknown_case.to_dict()],
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert (
        "unknown relevant chunk IDs"
        in captured.err
    )


def test_rejects_empty_case_file(
    tmp_path: Path,
    capsys,
):
    chunks_path, cases_path = (
        make_input_files(tmp_path)
    )
    cases_path.write_text(
        "",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "cases cannot be empty" in captured.err


def test_rejects_missing_chunk_file(
    tmp_path: Path,
    capsys,
):
    missing_chunks = (
        tmp_path / "missing.jsonl"
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--chunks",
            str(missing_chunks),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_rejects_non_positive_top_k(
    tmp_path: Path,
    capsys,
):
    chunks_path, cases_path = (
        make_input_files(tmp_path)
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
            "--top-k",
            "0",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert (
        "top_k must be at least 1"
        in captured.err
    )
