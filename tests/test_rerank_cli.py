from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.retrieval.rerank.cli import main
from rag_lab.vector_store import QdrantVectorStore
from tests.helpers import (
    FakeEmbeddingProvider,
    FakeVectorStore,
    make_chunk,
    write_chunks,
)


def test_writes_human_result_and_routes_configuration(
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
                index_text="TCP 使用拥塞控制保证网络性能",
            ),
            make_chunk(
                chunk_id="chunk-http",
                ordinal=2,
                index_text="HTTP 支持 Web 应用通信",
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="rerank-test",
        candidate_count=2,
        matches=[
            VectorMatch(
                chunk=make_chunk(
                    chunk_id="chunk-tcp",
                    ordinal=1,
                    index_text=(
                        "TCP 使用拥塞控制保证网络性能"
                    ),
                ),
                score=0.9,
            )
        ],
    )
    provider_config: list[dict[str, object]] = []
    store_config: list[dict[str, object]] = []

    def provider_factory(
        **kwargs: object,
    ) -> FakeEmbeddingProvider:
        provider_config.append(kwargs)
        return provider

    def store_factory(
        **kwargs: object,
    ) -> FakeVectorStore:
        store_config.append(kwargs)
        return store

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--query",
            "TCP",
            "--collection",
            "rerank-test",
            "--top-k",
            "3",
            "--fetch-k",
            "20",
            "--rrf-weight",
            "1.0",
            "--overlap-weight",
            "2.0",
            "--heading-weight",
            "0.5",
            "--url",
            "http://qdrant.test:6333",
            "--model",
            "fake-model",
            "--host",
            "http://ollama.test:11434",
            "--dimensions",
            "2",
            "--embedding-timeout-seconds",
            "12.5",
            "--qdrant-timeout-seconds",
            "9",
        ],
        provider_factory=provider_factory,
        store_factory=store_factory,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert provider_config == [
        {
            "model_name": "fake-model",
            "dimensions": 2,
            "host": "http://ollama.test:11434",
            "timeout_seconds": 12.5,
        }
    ]
    assert store_config == [
        {
            "url": "http://qdrant.test:6333",
            "collection_name": "rerank-test",
            "dimensions": 2,
            "timeout_seconds": 9,
        }
    ]
    assert provider.query_calls == ["TCP"]
    assert store.search_calls[0][1] == 10
    assert "Query: TCP" in captured.out
    assert "Retriever: hybrid+rerank" in captured.out
    assert "chunk_id=chunk-tcp" in captured.out


def test_writes_json_result(
    tmp_path: Path,
    capsys,
):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            make_chunk(
                chunk_id="chunk-json",
                ordinal=1,
                index_text="JSON 查询测试",
            )
        ],
    )
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="rerank-test",
        candidate_count=1,
        matches=[
            VectorMatch(
                chunk=make_chunk(
                    chunk_id="chunk-json",
                    ordinal=1,
                    index_text="JSON 查询测试",
                ),
                score=0.8,
            )
        ],
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--query",
            "JSON 查询",
            "--collection",
            "rerank-test",
            "--dimensions",
            "2",
            "--json",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["query"] == "JSON 查询"
    assert payload["retriever"] == "hybrid+rerank"
    assert payload["candidate_count"] >= 1


def test_missing_collection_returns_exit_two_without_creation(
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
    client = QdrantClient(":memory:")
    collection_name = "missing-search-rerank"
    provider = FakeEmbeddingProvider()
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        dimensions=2,
    )

    exit_code = main(
        [
            "--chunks",
            str(chunks_path),
            "--query",
            "TCP",
            "--collection",
            collection_name,
            "--dimensions",
            "2",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "does not exist" in captured.err
    assert client.collection_exists(collection_name) is False
