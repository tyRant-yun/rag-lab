from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.retrieval.dense import DenseRetriever


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


class RecordingEmbeddingProvider:
    def __init__(
        self,
        *,
        dimensions: int = 3,
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
            "DenseRetriever must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        self.query_calls.append(text)

        return EmbeddingVector(
            values=[0.6, 0.8, 0.1],
            dimensions=self.dimensions,
        )


class RecordingVectorStore:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        collection_name: str = "test-collection",
        candidate_count: int = 3,
        matches: Sequence[VectorMatch] = (),
    ) -> None:
        self._dimensions = dimensions
        self._collection_name = collection_name
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
        return None

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "DenseRetriever must not upsert records"
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


def test_search_embeds_counts_and_preserves_matches():
    first_chunk = make_chunk(
        chunk_id="chunk-first",
        ordinal=1,
    )
    second_chunk = make_chunk(
        chunk_id="chunk-second",
        ordinal=2,
    )
    matches = [
        VectorMatch(
            chunk=second_chunk,
            score=0.9,
        ),
        VectorMatch(
            chunk=first_chunk,
            score=0.8,
        ),
    ]
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore(
        candidate_count=7,
        matches=matches,
    )
    filters = SearchFilters(
        document_ids=["document-a"],
    )
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )

    result = retriever.search(
        "网络协议是什么？",
        top_k=4,
        filters=filters,
    )

    assert provider.query_calls == [
        "网络协议是什么？"
    ]
    assert store.count_calls == [filters]
    assert store.count_calls[0] is filters
    assert len(store.search_calls) == 1
    assert store.search_calls[0][0].values == [
        0.6,
        0.8,
        0.1,
    ]
    assert store.search_calls[0][1] == 4
    assert store.search_calls[0][2] is filters

    assert result.query == "网络协议是什么？"
    assert result.candidate_count == 7
    assert result.retriever == "dense"
    assert result.index_version == (
        "dense-v1:test-collection:"
        "fake:fake-model:3:v1"
    )
    assert [
        hit.chunk.chunk_id
        for hit in result.hits
    ] == [
        "chunk-second",
        "chunk-first",
    ]
    assert [
        hit.score
        for hit in result.hits
    ] == [0.9, 0.8]
    assert [
        hit.rank
        for hit in result.hits
    ] == [1, 2]
    assert all(
        hit.retriever == "dense"
        for hit in result.hits
    )


def test_rejects_dimension_mismatch():
    provider = RecordingEmbeddingProvider(
        dimensions=3,
    )
    store = RecordingVectorStore(
        dimensions=4,
    )

    with pytest.raises(
        ValueError,
        match=(
            "embedding provider dimensions must match "
            "vector store dimensions"
        ),
    ):
        DenseRetriever(
            provider=provider,
            store=store,
        )


def test_search_allows_fewer_matches_than_top_k():
    chunk = make_chunk(
        chunk_id="only-match",
        ordinal=1,
    )
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore(
        candidate_count=4,
        matches=[
            VectorMatch(
                chunk=chunk,
                score=0.7,
            )
        ],
    )
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )

    result = retriever.search(
        "一个查询",
        top_k=10,
    )

    assert result.candidate_count == 4
    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "only-match"
    ]
    assert store.search_calls[0][1] == 10


def test_search_returns_valid_empty_result():
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore(
        candidate_count=3,
    )
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )

    result = retriever.search("没有匹配")

    assert result.hits == []
    assert result.candidate_count == 3
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize(
    ("query", "top_k", "filters", "error_type", "message"),
    [
        (123, 5, None, TypeError, "query must be a string"),
        ("   ", 5, None, ValueError, "query cannot be empty"),
        ("query", True, None, TypeError, "top_k must be an integer"),
        ("query", 1.5, None, TypeError, "top_k must be an integer"),
        ("query", 0, None, ValueError, "top_k must be at least 1"),
        ("query", -1, None, ValueError, "top_k must be at least 1"),
        (
            "query",
            5,
            {"document_ids": ["document-a"]},
            TypeError,
            "filters must be SearchFilters",
        ),
    ],
)
def test_search_rejects_invalid_input(
    query: object,
    top_k: object,
    filters: object,
    error_type: type[Exception],
    message: str,
):
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore()
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )

    with pytest.raises(error_type, match=message):
        retriever.search(
            query,  # type: ignore[arg-type]
            top_k=top_k,  # type: ignore[arg-type]
            filters=filters,  # type: ignore[arg-type]
        )

    assert provider.query_calls == []
    assert store.count_calls == []
    assert store.search_calls == []


def test_search_propagates_provider_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore()
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )
    original_error = RuntimeError("embedding unavailable")

    def fail_embed_query(text: str) -> EmbeddingVector:
        del text
        raise original_error

    monkeypatch.setattr(
        provider,
        "embed_query",
        fail_embed_query,
    )

    with pytest.raises(RuntimeError) as error_info:
        retriever.search("一个查询")

    assert error_info.value is original_error
    assert store.count_calls == []
    assert store.search_calls == []


def test_search_propagates_store_count_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore()
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )
    original_error = RuntimeError("count unavailable")

    def fail_count(
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        del filters
        raise original_error

    monkeypatch.setattr(store, "count", fail_count)

    with pytest.raises(RuntimeError) as error_info:
        retriever.search("一个查询")

    assert error_info.value is original_error
    assert provider.query_calls == ["一个查询"]
    assert store.search_calls == []


def test_search_propagates_store_search_error(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = RecordingEmbeddingProvider()
    store = RecordingVectorStore()
    retriever = DenseRetriever(
        provider=provider,
        store=store,
    )
    original_error = RuntimeError("search unavailable")

    def fail_search(
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector, top_k, filters
        raise original_error

    monkeypatch.setattr(store, "search", fail_search)

    with pytest.raises(RuntimeError) as error_info:
        retriever.search("一个查询")

    assert error_info.value is original_error
    assert provider.query_calls == ["一个查询"]
    assert store.count_calls == [None]


def test_dense_retriever_is_publicly_exported():
    from rag_lab.retrieval import dense

    assert dense.DenseRetriever is DenseRetriever
    assert dense.__all__ == ["DenseRetriever"]
