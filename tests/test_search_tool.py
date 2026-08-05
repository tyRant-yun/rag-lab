from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.tools import (
    RetrievalToolset,
    SearchKnowledgeArguments,
    SearchKnowledgeTool,
)


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
        heading_path=["第一章", "1.1 什么是因特网"],
        page_start=ordinal,
        page_end=ordinal,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


class FakeRetriever:
    def __init__(
        self,
        *,
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


def make_result(
    *,
    retriever: str = "bm25",
) -> SearchResult:
    chunk = make_chunk(
        chunk_id="chunk-tcp",
        ordinal=1,
        index_text="TCP 使用拥塞控制保证网络性能",
    )
    return SearchResult(
        query="TCP",
        hits=[
            SearchHit(
                chunk=chunk,
                score=0.9,
                rank=1,
                retriever=retriever,
            )
        ],
        candidate_count=3,
        elapsed_ms=1.0,
        retriever=retriever,
        index_version="bm25-v1:index",
    )


def test_openai_schema_shape():
    schema = SearchKnowledgeTool.openai_schema()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "search_knowledge"
    assert "search" in function["description"].lower()
    parameters = function["parameters"]
    assert "query" in parameters["required"]
    assert parameters["properties"]["retriever"]["default"] == (
        "rerank"
    )


def test_execute_returns_bounded_hits():
    retriever = FakeRetriever(result=make_result())
    tool = SearchKnowledgeTool(
        retrievers={"bm25": retriever}
    )

    result = tool.execute(
        SearchKnowledgeArguments(
            query="TCP",
            retriever="bm25",
            top_k=3,
        )
    )

    assert result["success"] is True
    assert result["tool"] == "search_knowledge"
    assert result["retriever"] == "bm25"
    assert result["count"] == 1
    hit = result["hits"][0]
    assert hit["chunk_id"] == "chunk-tcp"
    assert hit["content"].startswith("TCP")
    assert hit["heading_path"] == [
        "第一章",
        "1.1 什么是因特网",
    ]
    assert hit["page_start"] == 1
    assert hit["source_path"] == "book.pdf"
    assert retriever.calls == [
        ("TCP", 3, SearchFilters())
    ]


def test_execute_raw_rejects_empty_query_without_search():
    retriever = FakeRetriever(result=make_result())
    tool = SearchKnowledgeTool(
        retrievers={"bm25": retriever}
    )

    result = tool.execute_raw({"query": ""})

    assert result["success"] is False
    assert "invalid arguments" in result["error"]
    assert retriever.calls == []


def test_execute_raw_rejects_unknown_retriever():
    retriever = FakeRetriever(result=make_result())
    tool = SearchKnowledgeTool(
        retrievers={"bm25": retriever}
    )

    result = tool.execute_raw(
        {
            "query": "TCP",
            "retriever": "unknown",
        }
    )

    assert result["success"] is False
    assert "invalid arguments" in result["error"]
    assert retriever.calls == []


def test_execute_passes_filters():
    retriever = FakeRetriever(result=make_result())
    tool = SearchKnowledgeTool(
        retrievers={"bm25": retriever}
    )
    filters = SearchFilters(
        document_ids=["document-a"],
        heading_prefix=["第一章"],
        page_start=1,
        page_end=2,
    )

    result = tool.execute(
        SearchKnowledgeArguments(
            query="TCP",
            retriever="bm25",
            document_ids=["document-a"],
            heading_prefix=["第一章"],
            page_start=1,
            page_end=2,
        )
    )

    assert result["success"] is True
    assert retriever.calls == [
        ("TCP", 5, filters)
    ]


def test_execute_rejects_empty_document_ids():
    retriever = FakeRetriever(result=make_result())
    tool = SearchKnowledgeTool(
        retrievers={"bm25": retriever}
    )

    result = tool.execute_raw(
        {
            "query": "TCP",
            "retriever": "bm25",
            "document_ids": [],
        }
    )

    assert result["success"] is False
    assert "document_ids" in result["error"]
    assert retriever.calls == []


class FakeEmbeddingProvider:
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def embedding_version(self) -> str:
        return "fake:fake-model:2:v1"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        del texts
        raise AssertionError(
            "tool search must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        del text
        raise AssertionError(
            "bm25 tool search must not embed queries"
        )


class FakeVectorStore:
    @property
    def collection_name(self) -> str:
        return "tools-test"

    @property
    def dimensions(self) -> int:
        return 2

    def ensure_collection(self) -> None:
        raise AssertionError(
            "tool search must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "tool search must not index records"
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        del filters
        return 1

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector, top_k, filters
        return []


def write_chunks(
    path: Path,
    chunks: Sequence[KnowledgeChunk],
) -> None:
    import json

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


def test_toolset_build_exposes_schema_and_executes(
    tmp_path: Path,
):
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
            )
        ],
    )
    toolset = RetrievalToolset.build(
        chunks_path=chunks_path,
        collection="tools-test",
        dimensions=2,
        provider_factory=lambda **_: FakeEmbeddingProvider(),
        store_factory=lambda **_: FakeVectorStore(),
    )

    assert toolset.to_openai_tools()[0][
        "function"
    ]["name"] == "search_knowledge"

    result = toolset.execute(
        {
            "query": "TCP",
            "retriever": "bm25",
        }
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["hits"][0]["chunk_id"] == (
        "chunk-tcp"
    )


def test_toolset_rejects_empty_chunks(
    tmp_path: Path,
):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="chunks cannot be empty",
    ):
        RetrievalToolset.build(
            chunks_path=empty_path,
            collection="tools-test",
            provider_factory=lambda **_: FakeEmbeddingProvider(),
            store_factory=lambda **_: FakeVectorStore(),
        )
