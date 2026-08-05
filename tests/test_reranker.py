from __future__ import annotations

import pytest

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchHit,
    SearchResult,
)
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)
from rag_lab.retrieval.rerank import (
    LexicalOverlapReranker,
)


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
    index_text: str,
    heading_path: list[str] | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=index_text,
        index_text=index_text,
        heading_path=heading_path or ["第一章"],
        page_start=ordinal,
        page_end=ordinal,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_result(
    *,
    query: str,
    hits: list[SearchHit],
    candidate_count: int,
    index_version: str = "hybrid-v1:index",
) -> SearchResult:
    return SearchResult(
        query=query,
        hits=hits,
        candidate_count=candidate_count,
        elapsed_ms=1.0,
        retriever="hybrid",
        index_version=index_version,
    )


def make_hit(
    chunk: KnowledgeChunk,
    rank: int,
    score: float,
) -> SearchHit:
    return SearchHit(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever="hybrid",
    )


def test_lexical_overlap_moves_relevant_chunk_to_top():
    analyzer = LexicalAnalyzer()
    reranker = LexicalOverlapReranker(analyzer=analyzer)
    relevant = make_chunk(
        chunk_id="chunk-a",
        ordinal=1,
        index_text="网络与因特网基础",
    )
    other = make_chunk(
        chunk_id="chunk-b",
        ordinal=2,
        index_text="协议与数据封装",
    )
    result = make_result(
        query="网络",
        hits=[
            make_hit(other, 1, 0.02),
            make_hit(relevant, 2, 0.02),
        ],
        candidate_count=10,
    )

    reranked = reranker.rerank(result, top_k=2)

    assert [
        hit.chunk.chunk_id
        for hit in reranked.hits
    ] == ["chunk-a", "chunk-b"]
    assert reranked.hits[0].score > reranked.hits[1].score
    assert reranked.retriever == "hybrid+rerank"
    assert reranked.index_version.endswith(
        "|rerank-v1:lexical-overlap"
    )
    assert reranked.candidate_count == 10
    assert [hit.rank for hit in reranked.hits] == [1, 2]


def test_heading_overlap_breaks_tie():
    analyzer = LexicalAnalyzer()
    reranker = LexicalOverlapReranker(analyzer=analyzer)
    no_heading = make_chunk(
        chunk_id="chunk-a",
        ordinal=1,
        index_text="网络基础",
        heading_path=["网络"],
    )
    with_heading = make_chunk(
        chunk_id="chunk-b",
        ordinal=2,
        index_text="网络基础",
        heading_path=["协议"],
    )
    result = make_result(
        query="协议",
        hits=[
            make_hit(no_heading, 1, 0.02),
            make_hit(with_heading, 2, 0.02),
        ],
        candidate_count=10,
    )

    reranked = reranker.rerank(result, top_k=2)

    assert [
        hit.chunk.chunk_id
        for hit in reranked.hits
    ] == ["chunk-b", "chunk-a"]


def test_respects_top_k_and_keeps_contiguous_ranks():
    analyzer = LexicalAnalyzer()
    reranker = LexicalOverlapReranker(analyzer=analyzer)
    hits = []
    for ordinal in range(1, 5):
        hits.append(
            make_hit(
                make_chunk(
                    chunk_id=f"chunk-{ordinal}",
                    ordinal=ordinal,
                    index_text=f"检索主题 {ordinal}",
                ),
                ordinal,
                0.01,
            )
        )
    result = make_result(
        query="检索",
        hits=hits,
        candidate_count=4,
    )

    reranked = reranker.rerank(result, top_k=2)

    assert len(reranked.hits) == 2
    assert [hit.rank for hit in reranked.hits] == [1, 2]
    assert reranked.candidate_count == 4


def test_deterministic_tie_break_by_chunk_id():
    analyzer = LexicalAnalyzer()
    reranker = LexicalOverlapReranker(analyzer=analyzer)
    chunk_b = make_chunk(
        chunk_id="chunk-b",
        ordinal=1,
        index_text="协议无关文本",
    )
    chunk_a = make_chunk(
        chunk_id="chunk-a",
        ordinal=2,
        index_text="协议无关文本",
    )
    result = make_result(
        query="协议",
        hits=[
            make_hit(chunk_b, 1, 0.02),
            make_hit(chunk_a, 2, 0.02),
        ],
        candidate_count=2,
    )

    reranked = reranker.rerank(result, top_k=2)

    assert [
        hit.chunk.chunk_id
        for hit in reranked.hits
    ] == ["chunk-a", "chunk-b"]


def test_empty_query_terms_returns_result_unchanged():
    analyzer = LexicalAnalyzer()
    reranker = LexicalOverlapReranker(analyzer=analyzer)
    chunk = make_chunk(
        chunk_id="chunk-a",
        ordinal=1,
        index_text="网络基础",
    )
    result = make_result(
        query="？？？",
        hits=[make_hit(chunk, 1, 0.02)],
        candidate_count=1,
    )

    reranked = reranker.rerank(result, top_k=1)

    assert reranked is result
    assert reranked.retriever == "hybrid"


def test_validation_errors():
    analyzer = LexicalAnalyzer()
    chunk = make_chunk(
        chunk_id="chunk-a",
        ordinal=1,
        index_text="网络基础",
    )
    result = make_result(
        query="网络",
        hits=[make_hit(chunk, 1, 0.02)],
        candidate_count=1,
    )
    reranker = LexicalOverlapReranker(
        analyzer=analyzer,
    )

    with pytest.raises(TypeError, match="top_k must be an integer"):
        reranker.rerank(result, top_k=1.5)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reranker.rerank(result, top_k=0)

    with pytest.raises(TypeError, match="result must be SearchResult"):
        reranker.rerank({}, top_k=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"overlap_weight": -1},
            ValueError,
            "overlap_weight must not be negative",
        ),
        (
            {"heading_weight": float("nan")},
            ValueError,
            "heading_weight must be finite",
        ),
        (
            {"rrf_weight": True},
            TypeError,
            "rrf_weight must be a number",
        ),
    ],
)
def test_invalid_weights(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
):
    with pytest.raises(error_type, match=message):
        LexicalOverlapReranker(
            analyzer=LexicalAnalyzer(),
            **kwargs,
        )
