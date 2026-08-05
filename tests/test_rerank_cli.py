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
    def __init__(
        self,
        *,
        dimensions: int = 2,
    ) -> None:
        self._dimensions = dimensions
        self.query_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def embedding_version(self) -> str:
        return (
            "fake:fake-model:"
            f"{self.dimensions}:v1"
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        del texts
        raise AssertionError(
            "search-rerank must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        self.query_calls.append(text)

        return EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=self.dimensions,
        )


class FakeVectorStore:
    def __init__(
        self,
        *,
        collection_name: str,
        dimensions: int = 2,
        candidate_count: int = 0,
        matches: Sequence[VectorMatch] = (),
    ) -> None:
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._candidate_count = candidate_count
        self._matches = list(matches)
        self.count_calls: list[SearchFilters | None] = []
        self.search_calls: list[
            tuple[
                EmbeddingVector,
                int,
                SearchFilters | None,
            ]
        ] = []

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def ensure_collection(self) -> None:
        raise AssertionError(
            "search-rerank must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "search-rerank must not index records"
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        self.count_calls.append(filters)
        return self._candidate_count

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        self.search_calls.append(
            (vector, top_k, filters)
        )
        return list(self._matches)


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
