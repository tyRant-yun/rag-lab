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
from tests.helpers import (
    FakeEmbeddingProvider,
    FakeVectorStore,
    make_chunk,
    write_chunks,
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


def make_client(
    tmp_path: Path,
    *,
    provider: FakeEmbeddingProvider | None = None,
    store: FakeVectorStore | None = None,
    enable_debug_routes: bool = True,
    readiness_checker=lambda: None,
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
        enable_debug_routes=enable_debug_routes,
        readiness_checker=readiness_checker,
    )
    return TestClient(app), chunks_path


def test_health(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_checks_dependencies_without_exposing_errors(
    tmp_path: Path,
):
    ready_client, _ = make_client(tmp_path)

    assert ready_client.get("/health/ready").json() == {
        "status": "ready"
    }

    def unavailable() -> None:
        raise RuntimeError("internal dependency address")

    unavailable_client, _ = make_client(
        tmp_path,
        readiness_checker=unavailable,
    )
    response = unavailable_client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "internal dependency" not in response.text


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
    assert payload["hits"][0]["chunk_id"] == (
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
    assert payload["hits"][0]["chunk_id"] == (
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


def test_public_search_hides_internal_retrieval_fields(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/search",
        json={"query": "TCP"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"request_id", "results"}
    assert payload["request_id"].startswith("req_")
    assert set(payload["results"][0]) == {"content", "citation"}
    assert set(payload["results"][0]["citation"]) == {
        "title",
        "section",
        "pages",
    }
    assert "source_path" not in response.text
    assert "chunk_id" not in response.text
    assert "index_version" not in response.text


def test_public_search_rejects_retriever_controls(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/search",
        json={"query": "TCP", "retriever": "bm25"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_public_knowledge_base_exposes_only_product_metadata(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    response = client.get("/api/v1/knowledge-base")

    assert response.status_code == 200
    assert set(response.json()) == {
        "title", "coverage", "topics", "capabilities", "guidance", "limitations"
    }
    assert "collection" not in response.text
    assert "source_path" not in response.text


def test_public_search_checks_received_body_size(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/search",
        content=b'{"query":"' + b"x" * 9000 + b'"}',
        headers={
            "content-type": "application/json",
            "content-length": "0",
        },
    )

    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"


def test_public_mode_hides_debug_routes(tmp_path: Path):
    client, _ = make_client(tmp_path, enable_debug_routes=False)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.post("/search", json={"query": "TCP"}).status_code == 404


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
    assert response.json()["hits"][0]["chunk_id"] == (
        "chunk-tcp"
    )


def test_source_path_hidden_by_default_and_opt_in(
    tmp_path: Path,
):
    client, _ = make_client(tmp_path)

    hidden = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "bm25",
        },
    ).json()

    assert hidden["hits"][0]["source_path"] is None

    shown = client.post(
        "/search",
        json={
            "query": "TCP",
            "retriever": "bm25",
            "include_source_path": True,
        },
    ).json()

    assert shown["hits"][0]["source_path"] == "book.pdf"


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
