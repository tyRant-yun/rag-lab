from __future__ import annotations

import json
from collections.abc import Sequence

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
from rag_lab.retrieval.dense.cli import main
from rag_lab.vector_store import QdrantVectorStore


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=f"正文 {chunk_id}",
        index_text=f"索引正文 {chunk_id}",
        heading_path=["第一章", "运输层"],
        page_start=19,
        page_end=22,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
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
            "search-dense must not embed documents"
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
            "search-dense must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "search-dense must not index records"
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


def test_writes_human_result_and_routes_all_configuration(
    capsys,
):
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="dense-test",
        candidate_count=4,
        matches=[
            VectorMatch(
                chunk=make_chunk(
                    chunk_id="chunk-tcp",
                    ordinal=1,
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
            "--collection",
            "dense-test",
            "--query",
            "TCP 如何提供可靠传输？",
            "--top-k",
            "3",
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
            "--document-id",
            "document-a",
            "--heading",
            "第一章",
            "--heading",
            "运输层",
            "--page-start",
            "20",
            "--page-end",
            "24",
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
            "collection_name": "dense-test",
            "dimensions": 2,
            "timeout_seconds": 9,
        }
    ]
    assert provider.query_calls == [
        "TCP 如何提供可靠传输？"
    ]
    assert store.count_calls == [
        SearchFilters(
            document_ids=["document-a"],
            heading_prefix=["第一章", "运输层"],
            page_start=20,
            page_end=24,
        )
    ]
    assert len(store.search_calls) == 1
    assert store.search_calls[0][0].values == [0.6, 0.8]
    assert store.search_calls[0][1] == 3
    assert (
        store.search_calls[0][2]
        is store.count_calls[0]
    )
    assert "Query: TCP 如何提供可靠传输？" in captured.out
    assert "Retriever: dense" in captured.out
    assert "Candidates: 4" in captured.out
    assert "chunk_id=chunk-tcp" in captured.out


def test_writes_json_result(
    capsys,
):
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="dense-test",
        candidate_count=2,
        matches=[
            VectorMatch(
                chunk=make_chunk(
                    chunk_id="chunk-json",
                    ordinal=1,
                ),
                score=0.8,
            )
        ],
    )

    exit_code = main(
        [
            "--collection",
            "dense-test",
            "--query",
            "JSON 查询",
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
    assert payload["retriever"] == "dense"
    assert payload["candidate_count"] == 2
    assert payload["hits"][0]["chunk"]["chunk_id"] == (
        "chunk-json"
    )


def test_reports_empty_result(
    capsys,
):
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="dense-test",
        candidate_count=2,
    )

    exit_code = main(
        [
            "--collection",
            "dense-test",
            "--query",
            "没有匹配",
            "--dimensions",
            "2",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Candidates: 2" in captured.out
    assert "Hits: 0" in captured.out
    assert "No matching chunks." in captured.out


def test_reports_invalid_filter_without_retrieval(
    capsys,
):
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(
        collection_name="dense-test",
    )

    exit_code = main(
        [
            "--collection",
            "dense-test",
            "--query",
            "TCP",
            "--dimensions",
            "2",
            "--page-start",
            "0",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "page_start must be at least 1" in captured.err
    assert provider.query_calls == []
    assert store.count_calls == []
    assert store.search_calls == []


def test_missing_collection_returns_exit_two_without_creation(
    capsys,
):
    client = QdrantClient(":memory:")
    collection_name = "missing-search-dense"
    provider = FakeEmbeddingProvider()
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        dimensions=2,
    )

    exit_code = main(
        [
            "--collection",
            collection_name,
            "--query",
            "TCP",
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
