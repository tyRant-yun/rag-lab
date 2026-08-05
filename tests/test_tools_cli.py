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
from rag_lab.tools.cli import (
    execute_main,
    schema_main,
)
from tests.helpers import (
    FakeEmbeddingProvider,
    FakeVectorStore,
    make_chunk,
    write_chunks,
)


def test_schema_main_prints_openai_schema(capsys):
    exit_code = schema_main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["function"]["name"] == (
        "search_knowledge"
    )
    assert "query" in payload["function"]["parameters"][
        "required"
    ]


def test_execute_main_runs_tool(
    tmp_path: Path,
    capsys,
):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            make_chunk(
                chunk_id="chunk-tcp",
                ordinal=1,
                index_text=(
                    "TCP 使用拥塞控制保证网络性能"
                ),
            )
        ],
    )

    exit_code = execute_main(
        [
            "--chunks",
            str(chunks_path),
            "--collection",
            "tools-cli",
            "--dimensions",
            "2",
            "--args",
            (
                '{"query": "TCP", '
                '"retriever": "bm25"}'
            ),
        ],
        provider_factory=lambda **_: FakeEmbeddingProvider(),
        store_factory=lambda **_: FakeVectorStore(
            collection_name="tools-cli"
        ),
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["success"] is True
    assert payload["tool"] == "search_knowledge"
    assert payload["count"] == 1
    assert payload["hits"][0]["chunk_id"] == (
        "chunk-tcp"
    )


def test_execute_main_rejects_invalid_json(
    tmp_path: Path,
    capsys,
):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            make_chunk(
                chunk_id="chunk-tcp",
                ordinal=1,
                index_text="TCP",
            )
        ],
    )

    exit_code = execute_main(
        [
            "--chunks",
            str(chunks_path),
            "--collection",
            "tools-cli",
            "--dimensions",
            "2",
            "--args",
            "not-json",
        ],
        provider_factory=lambda **_: FakeEmbeddingProvider(),
        store_factory=lambda **_: FakeVectorStore(),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "error:" in captured.err


def test_execute_main_rejects_non_object_args(
    tmp_path: Path,
    capsys,
):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            make_chunk(
                chunk_id="chunk-tcp",
                ordinal=1,
                index_text="TCP",
            )
        ],
    )

    exit_code = execute_main(
        [
            "--chunks",
            str(chunks_path),
            "--collection",
            "tools-cli",
            "--dimensions",
            "2",
            "--args",
            "[1, 2]",
        ],
        provider_factory=lambda **_: FakeEmbeddingProvider(),
        store_factory=lambda **_: FakeVectorStore(),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must be a JSON object" in captured.err
