import pytest

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
)
from rag_lab.retrieval.bm25 import (
    BM25Index,
    BM25Retriever,
)
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)


def make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    index_text: str,
    ordinal: int,
    heading_path: list[str],
    page_start: int,
    page_end: int | None = None,
) -> KnowledgeChunk:
    resolved_page_end = (
        page_start
        if page_end is None
        else page_end
    )

    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=index_text,
        index_text=index_text,
        heading_path=heading_path,
        page_start=page_start,
        page_end=resolved_page_end,
        ordinal=ordinal,
        block_ids=[f"block-{chunk_id}"],
        source_path="book.pdf",
        content_hash=f"hash-{chunk_id}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def make_chunks() -> tuple[KnowledgeChunk, ...]:
    return (
        make_chunk(
            chunk_id="chunk-tcp",
            document_id="document-a",
            index_text=(
                "TCP 使用拥塞控制保证网络性能"
            ),
            ordinal=1,
            heading_path=["第一章", "运输层"],
            page_start=10,
        ),
        make_chunk(
            chunk_id="chunk-http",
            document_id="document-a",
            index_text=(
                "HTTP 支持 Web 网络应用通信"
            ),
            ordinal=2,
            heading_path=["第二章", "应用层"],
            page_start=20,
            page_end=22,
        ),
        make_chunk(
            chunk_id="chunk-dns",
            document_id="document-b",
            index_text=(
                "DNS 将网络域名转换成 IP 地址"
            ),
            ordinal=3,
            heading_path=["第二章", "应用层"],
            page_start=24,
        ),
    )


def make_retriever() -> BM25Retriever:
    analyzer = LexicalAnalyzer(
        user_words=("拥塞控制",),
    )
    index = BM25Index(
        chunks=make_chunks(),
        analyzer=analyzer,
    )

    return BM25Retriever(index=index)


def test_returns_ranked_search_result():
    retriever = make_retriever()

    result = retriever.search(
        "TCP 拥塞控制"
    )

    assert result.query == "TCP 拥塞控制"
    assert result.candidate_count == 3
    assert result.retriever == "bm25"
    assert result.index_version.startswith(
        "bm25-v1:"
    )

    assert len(result.hits) == 1
    assert result.hits[0].rank == 1
    assert (
        result.hits[0].chunk.chunk_id
        == "chunk-tcp"
    )


def test_applies_document_filter():
    retriever = make_retriever()

    result = retriever.search(
        "DNS",
        filters=SearchFilters(
            document_ids=["document-b"],
        ),
    )

    assert result.candidate_count == 1
    assert len(result.hits) == 1
    assert (
        result.hits[0].chunk.document_id
        == "document-b"
    )


def test_applies_heading_prefix_filter():
    retriever = make_retriever()

    result = retriever.search(
        "HTTP DNS",
        filters=SearchFilters(
            heading_prefix=[
                "第二章",
                "应用层",
            ],
        ),
    )

    assert result.candidate_count == 2
    assert all(
        hit.chunk.heading_path[:2]
        == ["第二章", "应用层"]
        for hit in result.hits
    )


def test_page_filter_uses_interval_overlap():
    retriever = make_retriever()

    result = retriever.search(
        "HTTP",
        filters=SearchFilters(
            page_start=21,
            page_end=21,
        ),
    )

    assert result.candidate_count == 1
    assert len(result.hits) == 1
    assert (
        result.hits[0].chunk.chunk_id
        == "chunk-http"
    )


def test_respects_top_k():
    retriever = make_retriever()

    result = retriever.search(
        "网络",
        top_k=2,
    )

    assert len(result.hits) == 2
    assert [
        hit.rank
        for hit in result.hits
    ] == [1, 2]


def test_returns_empty_hits_for_unmatched_query():
    retriever = make_retriever()

    result = retriever.search(
        "量子纠缠"
    )

    assert result.candidate_count == 3
    assert result.hits == []


def test_keeps_exact_match_with_zero_bm25_score():
    chunks = (
        make_chunk(
            chunk_id="chunk-tcp",
            document_id="document-a",
            index_text="TCP",
            ordinal=1,
            heading_path=["第一章"],
            page_start=1,
        ),
        make_chunk(
            chunk_id="chunk-http",
            document_id="document-a",
            index_text="HTTP",
            ordinal=2,
            heading_path=["第一章"],
            page_start=2,
        ),
    )
    analyzer = LexicalAnalyzer()
    index = BM25Index(
        chunks=chunks,
        analyzer=analyzer,
    )
    retriever = BM25Retriever(index=index)

    result = retriever.search("TCP")

    assert result.candidate_count == 2
    assert len(result.hits) == 1
    assert (
        result.hits[0].chunk.chunk_id
        == "chunk-tcp"
    )
    assert result.hits[0].score == 0.0


def test_keeps_negative_matches_in_stable_order():
    chunks = (
        make_chunk(
            chunk_id="chunk-first",
            document_id="document-a",
            index_text="网络",
            ordinal=1,
            heading_path=["第一章"],
            page_start=1,
        ),
        make_chunk(
            chunk_id="chunk-second",
            document_id="document-a",
            index_text="网络",
            ordinal=2,
            heading_path=["第一章"],
            page_start=2,
        ),
    )
    analyzer = LexicalAnalyzer()
    index = BM25Index(
        chunks=chunks,
        analyzer=analyzer,
    )
    retriever = BM25Retriever(index=index)

    result = retriever.search("网络")

    assert [
        hit.chunk.chunk_id
        for hit in result.hits
    ] == [
        "chunk-first",
        "chunk-second",
    ]
    assert all(
        hit.score < 0.0
        for hit in result.hits
    )


def test_rejects_empty_query():
    retriever = make_retriever()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.search("   ")


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_rejects_non_positive_top_k(
    top_k: int,
):
    retriever = make_retriever()

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        retriever.search(
            "TCP",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "top_k",
    [True, 1.5],
)
def test_rejects_invalid_top_k_type(
    top_k: object,
):
    retriever = make_retriever()

    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        retriever.search(
            "TCP",
            top_k=top_k,  # type: ignore[arg-type]
        )
