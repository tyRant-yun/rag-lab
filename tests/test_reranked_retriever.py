from __future__ import annotations

import pytest

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.retrieval.rerank import (
    RerankedRetriever,
)


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
        page_start=1,
        page_end=2,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_result() -> SearchResult:
    chunk = make_chunk(chunk_id="chunk-a", ordinal=1)
    return SearchResult(
        query="query",
        hits=[
            SearchHit(
                chunk=chunk,
                score=0.9,
                rank=1,
                retriever="hybrid",
            )
        ],
        candidate_count=5,
        elapsed_ms=1.0,
        retriever="hybrid",
        index_version="hybrid-v1:index",
    )


class FakeRetriever:
    def __init__(
        self,
        result: SearchResult,
    ) -> None:
        self._result = result
        self.calls: list[
            tuple[
                str,
                int,
                SearchFilters | None,
            ]
        ] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        self.calls.append((query, top_k, filters))
        return self._result


class FakeReranker:
    def __init__(
        self,
        result: SearchResult,
    ) -> None:
        self._result = result
        self.calls: list[
            tuple[
                SearchResult,
                int,
            ]
        ] = []

    def rerank(
        self,
        result: SearchResult,
        *,
        top_k: int,
    ) -> SearchResult:
        self.calls.append((result, top_k))
        return self._result


def test_fetches_wider_window_then_reranks_to_top_k():
    base_result = make_result()
    reranked_result = make_result()
    retriever = FakeRetriever(base_result)
    reranker = FakeReranker(reranked_result)
    filters = SearchFilters(
        document_ids=["document-a"],
    )
    wrapper = RerankedRetriever(
        retriever=retriever,
        reranker=reranker,
        fetch_k=20,
    )

    result = wrapper.search(
        "query",
        top_k=3,
        filters=filters,
    )

    assert result is reranked_result
    assert retriever.calls == [
        ("query", 20, filters)
    ]
    assert reranker.calls == [(base_result, 3)]


def test_default_fetch_k_is_twenty():
    base_result = make_result()
    retriever = FakeRetriever(base_result)
    wrapper = RerankedRetriever(
        retriever=retriever,
        reranker=FakeReranker(base_result),
    )

    wrapper.search("query", top_k=1)

    assert retriever.calls == [("query", 20, None)]


def test_validation_errors():
    result = make_result()
    retriever = FakeRetriever(result)
    reranker = FakeReranker(result)

    with pytest.raises(ValueError, match="fetch_k must be at least 1"):
        RerankedRetriever(
            retriever=retriever,
            reranker=reranker,
            fetch_k=0,
        )

    with pytest.raises(TypeError, match="fetch_k must be an integer"):
        RerankedRetriever(
            retriever=retriever,
            reranker=reranker,
            fetch_k=True,
        )
