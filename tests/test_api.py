from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag_lab.api.app import create_app
from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
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
            "API search must not embed documents"
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


class LocalModeEmbeddingProvider(
    FakeEmbeddingProvider
):
    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        del text
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
            "API search must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "API search must not index records"
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


def make_client(
    tmp_path: Path,
    *,
    provider: FakeEmbeddingProvider | None = None,
    store: FakeVectorStore | None = None,
) -> tuple[TestClient, Path]:
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
            ),
            make_chunk(
                chunk_id="chunk-http",
                ordinal=2,
                index_text="HTTP 支持 Web 应用通信",
            ),
            make_chunk(
                chunk_id="chunk-dns",
                ordinal=3,
                index_text="DNS 将域名转换为 IP 地址",
            ),
        ],
    )
    active_provider = provider or FakeEmbeddingProvider()
    active_store = store or FakeVectorStore(
        collection_name="api-test",
        candidate_count=3,
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
    app = create_app(
        chunks_path=chunks_path,
        collection="api-test",
        dimensions=2,
        provider_factory=lambda **_: active_provider,
        store_factory=lambda **_: active_store,
    )
    return TestClient(app), chunks_path


def test_health(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_bm25(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "bm25",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "bm25"
    assert payload["hits"][0]["chunk"]["chunk_id"] == (
        "chunk-tcp"
    )


def test_search_dense_routes_to_vector_store(
    tmp_path: Path,
):
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="api-test",
        candidate_count=3,
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
    client, _ = make_client(
        tmp_path,
        provider=provider,
        store=store,
    )

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "dense",
            "top_k": 3,
            "page_start": 1,
            "page_end": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "dense"
    assert payload["hits"][0]["chunk"]["chunk_id"] == (
        "chunk-tcp"
    )
    assert provider.query_calls == ["TCP"]
    assert store.count_calls[0].page_start == 1
    assert store.count_calls[0].page_end == 3


def test_search_hybrid(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "hybrid",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "hybrid"
    assert payload["candidate_count"] >= 1


def test_search_defaults_to_rerank(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={"query": "TCP"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "hybrid+rerank"
    assert payload["hits"][0]["retriever"] == (
        "hybrid+rerank"
    )


def test_search_with_filters(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "bm25",
            "document_ids": ["document-a"],
            "heading_prefix": ["第一章"],
        },
    )

    assert response.status_code == 200
    assert response.json()["hits"][0]["chunk"][
        "chunk_id"
    ] == "chunk-tcp"


def test_invalid_retriever_name_returns_422(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "unknown",
        },
    )

    assert response.status_code == 422


def test_empty_query_returns_422(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_invalid_page_range_returns_422(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "page_start": 5,
            "page_end": 3,
        },
    )

    assert response.status_code == 422
    assert "page_end must not precede page_start" in (
        response.text
    )


def test_upstream_error_maps_to_502(
    tmp_path: Path,
):
    client_obj = QdrantClient(":memory:")
    collection_name = "missing-api-collection"
    provider = LocalModeEmbeddingProvider()
    store = QdrantVectorStore(
        client=client_obj,
        collection_name=collection_name,
        dimensions=2,
    )
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
    app = create_app(
        chunks_path=chunks_path,
        collection=collection_name,
        dimensions=2,
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )
    client = TestClient(app)

    response = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "dense",
        },
    )

    assert response.status_code == 502
    assert "does not exist" in response.text


def test_create_app_rejects_empty_chunks(
    tmp_path: Path,
):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunks cannot be empty"):
        create_app(
            chunks_path=empty_path,
            collection="api-test",
            provider_factory=lambda **_: FakeEmbeddingProvider(),
            store_factory=lambda **_: FakeVectorStore(
                collection_name="api-test",
            ),
        )
