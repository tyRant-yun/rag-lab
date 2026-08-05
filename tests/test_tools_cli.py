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


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
    index_text: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=index_text or f"正文 {chunk_id}",
        index_text=index_text or f"索引正文 {chunk_id}",
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
    chunks: Sequence[KnowledgeChunk],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                chunk.to_dict(),
                ensure_ascii=False,
            )
            for chunk in chunks
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class FakeEmbeddingProvider:
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
        del texts
        raise AssertionError(
            "tool CLI must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        del text
        raise AssertionError(
            "bm25 tool CLI must not embed queries"
        )


class FakeVectorStore:
    @property
    def collection_name(self) -> str:
        return "tools-cli"

    @property
    def dimensions(self) -> int:
        return 2

    def ensure_collection(self) -> None:
        raise AssertionError(
            "tool CLI must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "tool CLI must not index records"
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        del filters
        return 1

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector, top_k, filters
        return []


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
        store_factory=lambda **_: FakeVectorStore(),
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
