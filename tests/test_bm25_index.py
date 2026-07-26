import pytest

from rag_lab.contracts import KnowledgeChunk
from rag_lab.retrieval.bm25 import BM25Index
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)


def make_chunk(
    *,
    chunk_id: str,
    index_text: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        content=index_text,
        index_text=index_text,
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


def make_chunks() -> tuple[KnowledgeChunk, ...]:
    return (
        make_chunk(
            chunk_id="chunk-tcp",
            index_text="TCP 使用拥塞控制保证网络性能",
            ordinal=1,
        ),
        make_chunk(
            chunk_id="chunk-http",
            index_text="HTTP 支持 Web 应用通信",
            ordinal=2,
        ),
        make_chunk(
            chunk_id="chunk-dns",
            index_text="DNS 将域名转换成 IP 地址",
            ordinal=3,
        ),
    )


def make_analyzer() -> LexicalAnalyzer:
    return LexicalAnalyzer(
        user_words=("拥塞控制",),
    )


def test_builds_index_in_chunk_order():
    chunks = make_chunks()

    index = BM25Index(
        chunks=chunks,
        analyzer=make_analyzer(),
    )

    assert index.chunks == chunks
    assert index.size == 3
    assert len(index.tokenized_corpus) == 3


def test_scores_relevant_chunk_higher():
    index = BM25Index(
        chunks=make_chunks(),
        analyzer=make_analyzer(),
    )

    scores = index.score_query(
        "TCP 拥塞控制"
    )

    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_blank_query_returns_zero_scores():
    index = BM25Index(
        chunks=make_chunks(),
        analyzer=make_analyzer(),
    )

    assert index.score_query("   ") == (
        0.0,
        0.0,
        0.0,
    )


def test_score_terms_matches_score_query():
    index = BM25Index(
        chunks=make_chunks(),
        analyzer=make_analyzer(),
    )

    query_terms = index.analyze_query(
        "TCP 拥塞控制"
    )

    assert index.score_terms(
        query_terms
    ) == index.score_query(
        "TCP 拥塞控制"
    )


def test_rejects_empty_chunk_collection():
    with pytest.raises(
        ValueError,
        match="chunks cannot be empty",
    ):
        BM25Index(
            chunks=(),
            analyzer=make_analyzer(),
        )


def test_rejects_duplicate_chunk_ids():
    chunk = make_chunk(
        chunk_id="duplicate",
        index_text="网络协议",
        ordinal=1,
    )

    with pytest.raises(
        ValueError,
        match="chunk IDs cannot contain duplicates",
    ):
        BM25Index(
            chunks=(chunk, chunk),
            analyzer=make_analyzer(),
        )


def test_rejects_chunk_without_lexical_terms():
    chunk = make_chunk(
        chunk_id="punctuation-only",
        index_text="。。。！！！",
        ordinal=1,
    )

    with pytest.raises(
        ValueError,
        match="produced no lexical terms",
    ):
        BM25Index(
            chunks=(chunk,),
            analyzer=make_analyzer(),
        )


def test_index_version_is_deterministic():
    first = BM25Index(
        chunks=make_chunks(),
        analyzer=make_analyzer(),
    )
    second = BM25Index(
        chunks=make_chunks(),
        analyzer=make_analyzer(),
    )

    assert first.index_version == second.index_version
    assert first.index_version.startswith(
        "bm25-v1:"
    )
