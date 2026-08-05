from __future__ import annotations

import pytest

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.retrieval.hybrid import (
    HybridRetriever,
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


def make_hit(
    chunk: KnowledgeChunk,
    rank: int,
    retriever: str,
) -> SearchHit:
    return SearchHit(
        chunk=chunk,
        score=1.0,
        rank=rank,
        retriever=retriever,
    )


def make_result(
    *,
    retriever: str,
    hits: list[SearchHit],
    candidate_count: int | None = None,
    index_version: str = "index-v1",
) -> SearchResult:
    return SearchResult(
        query="query",
        hits=hits,
        candidate_count=(
            candidate_count
            if candidate_count is not None
            else len(hits)
        ),
        elapsed_ms=1.0,
        retriever=retriever,
        index_version=index_version,
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


def build_retriever(
    *,
    bm25_result: SearchResult,
    dense_result: SearchResult,
    rrf_k: int = 60,
    per_retriever_k: int = 10,
) -> tuple[HybridRetriever, FakeRetriever, FakeRetriever]:
    bm25 = FakeRetriever(bm25_result)
    dense = FakeRetriever(dense_result)
    retriever = HybridRetriever(
        bm25=bm25,
        dense=dense,
        rrf_k=rrf_k,
        per_retriever_k=per_retriever_k,
    )
    return retriever, bm25, dense


def test_rrf_fuses_and_orders_by_fused_score():
    chunk_a = make_chunk(chunk_id="chunk-a", ordinal=1)
    chunk_b = make_chunk(chunk_id="chunk-b", ordinal=2)
    chunk_c = make_chunk(chunk_id="chunk-c", ordinal=3)
    chunk_d = make_chunk(chunk_id="chunk-d", ordinal=4)

    bm25_result = make_result(
        retriever="bm25",
        hits=[
            make_hit(chunk_a, 1, "bm25"),
            make_hit(chunk_b, 2, "bm25"),
            make_hit(chunk_c, 3, "bm25"),
        ],
        candidate_count=3,
        index_version="bm25-index",
    )
    dense_result = make_result(
        retriever="dense",
        hits=[
            make_hit(chunk_b, 1, "dense"),
            make_hit(chunk_d, 2, "dense"),
        ],
        candidate_count=2,
        index_version="dense-index",
    )

    retriever, bm25, dense = build_retriever(
        bm25_result=bm25_result,
        dense_result=dense_result,
    )

    result = retriever.search("query", top_k=5)

    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "chunk-b",
        "chunk-a",
        "chunk-d",
        "chunk-c",
    ]
    assert [hit.rank for hit in result.hits] == [1, 2, 3, 4]
    assert result.hits[0].score == pytest.approx(
        1.0 / 61.0 + 1.0 / 62.0
    )
    assert result.hits[1].score == pytest.approx(1.0 / 61.0)
    assert result.hits[2].score == pytest.approx(1.0 / 62.0)
    assert result.hits[3].score == pytest.approx(1.0 / 63.0)
    assert all(
        hit.retriever == "hybrid"
        for hit in result.hits
    )
    assert result.retriever == "hybrid"
    assert result.candidate_count == 4
    assert result.index_version == (
        "hybrid-v1:bm25-index|dense-index"
    )
    assert bm25.calls == [("query", 10, SearchFilters())]
    assert dense.calls == [("query", 10, SearchFilters())]


def test_respects_top_k():
    chunk_a = make_chunk(chunk_id="chunk-a", ordinal=1)
    chunk_b = make_chunk(chunk_id="chunk-b", ordinal=2)
    chunk_c = make_chunk(chunk_id="chunk-c", ordinal=3)

    bm25_result = make_result(
        retriever="bm25",
        hits=[
            make_hit(chunk_a, 1, "bm25"),
            make_hit(chunk_b, 2, "bm25"),
            make_hit(chunk_c, 3, "bm25"),
        ],
    )
    dense_result = make_result(
        retriever="dense",
        hits=[
            make_hit(chunk_a, 1, "dense"),
        ],
    )

    retriever, _, _ = build_retriever(
        bm25_result=bm25_result,
        dense_result=dense_result,
    )

    result = retriever.search("query", top_k=2)

    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "chunk-a",
        "chunk-b",
    ]
    assert result.candidate_count == 3


def test_passes_filters_to_both_retrievers():
    chunk = make_chunk(chunk_id="chunk-a", ordinal=1)
    filters = SearchFilters(
        document_ids=["document-a"],
        heading_prefix=["第一章"],
        page_start=1,
        page_end=2,
    )
    bm25_result = make_result(
        retriever="bm25",
        hits=[make_hit(chunk, 1, "bm25")],
    )
    dense_result = make_result(
        retriever="dense",
        hits=[make_hit(chunk, 1, "dense")],
    )

    retriever, bm25, dense = build_retriever(
        bm25_result=bm25_result,
        dense_result=dense_result,
    )

    result = retriever.search(
        "query",
        top_k=1,
        filters=filters,
    )

    assert result.hits[0].chunk.chunk_id == "chunk-a"
    assert bm25.calls == [("query", 10, filters)]
    assert dense.calls == [("query", 10, filters)]


def test_deterministic_tie_break_by_chunk_id():
    chunk_b = make_chunk(chunk_id="chunk-b", ordinal=1)
    chunk_a = make_chunk(chunk_id="chunk-a", ordinal=2)

    bm25_result = make_result(
        retriever="bm25",
        hits=[make_hit(chunk_a, 1, "bm25")],
    )
    dense_result = make_result(
        retriever="dense",
        hits=[make_hit(chunk_b, 1, "dense")],
    )

    retriever, _, _ = build_retriever(
        bm25_result=bm25_result,
        dense_result=dense_result,
    )

    result = retriever.search("query", top_k=2)

    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "chunk-a",
        "chunk-b",
    ]


def test_candidate_count_is_fused_union():
    chunk_a = make_chunk(chunk_id="chunk-a", ordinal=1)
    chunk_b = make_chunk(chunk_id="chunk-b", ordinal=2)
    chunk_c = make_chunk(chunk_id="chunk-c", ordinal=3)

    bm25_result = make_result(
        retriever="bm25",
        hits=[
            make_hit(chunk_a, 1, "bm25"),
            make_hit(chunk_b, 2, "bm25"),
        ],
    )
    dense_result = make_result(
        retriever="dense",
        hits=[
            make_hit(chunk_b, 1, "dense"),
            make_hit(chunk_c, 2, "dense"),
        ],
    )

    retriever, _, _ = build_retriever(
        bm25_result=bm25_result,
        dense_result=dense_result,
    )

    result = retriever.search("query", top_k=5)

    assert result.candidate_count == 3
    assert len(result.hits) == 3


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"rrf_k": 0},
            ValueError,
            "rrf_k must be at least 1",
        ),
        (
            {"rrf_k": True},
            TypeError,
            "rrf_k must be an integer",
        ),
        (
            {"per_retriever_k": 0},
            ValueError,
            "per_retriever_k must be at least 1",
        ),
        (
            {"per_retriever_k": 1.5},
            TypeError,
            "per_retriever_k must be an integer",
        ),
    ],
)
def test_invalid_constructor_parameters(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
):
    chunk = make_chunk(chunk_id="chunk-a", ordinal=1)
    result = make_result(
        retriever="bm25",
        hits=[make_hit(chunk, 1, "bm25")],
    )
    bm25 = FakeRetriever(result)
    dense = FakeRetriever(result)

    with pytest.raises(error_type, match=message):
        HybridRetriever(
            bm25=bm25,
            dense=dense,
            **kwargs,
        )


def test_invalid_search_parameters():
    chunk = make_chunk(chunk_id="chunk-a", ordinal=1)
    result = make_result(
        retriever="bm25",
        hits=[make_hit(chunk, 1, "bm25")],
    )
    retriever, _, _ = build_retriever(
        bm25_result=result,
        dense_result=result,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.search("")

    with pytest.raises(TypeError, match="top_k must be an integer"):
        retriever.search("query", top_k=1.5)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("query", top_k=0)

    with pytest.raises(TypeError, match="filters must be SearchFilters"):
        retriever.search("query", filters={})
