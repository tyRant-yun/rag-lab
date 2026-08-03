from __future__ import annotations

import pytest

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        content=f"content-{chunk_id}",
        index_text=f"index-{chunk_id}",
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


class FakeRetriever:
    def __init__(
        self,
        *,
        results_by_query: dict[
            str,
            tuple[KnowledgeChunk, ...],
        ],
        metadata_by_query: dict[
            str,
            tuple[str, str],
        ]
        | None = None,
    ) -> None:
        self._results_by_query = (
            results_by_query
        )
        self._metadata_by_query = (
            metadata_by_query or {}
        )
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
        self.calls.append(
            (
                query,
                top_k,
                filters,
            )
        )

        chunks = self._results_by_query[
            query
        ][:top_k]

        retriever_name, index_version = (
            self._metadata_by_query.get(
                query,
                (
                    "fake",
                    "fake-index-v1",
                ),
            )
        )

        hits = [
            SearchHit(
                chunk=chunk,
                score=float(
                    len(chunks) - position
                ),
                rank=position + 1,
                retriever=retriever_name,
            )
            for position, chunk in enumerate(
                chunks
            )
        ]

        return SearchResult(
            query=query,
            hits=hits,
            candidate_count=len(chunks),
            elapsed_ms=0.0,
            retriever=retriever_name,
            index_version=index_version,
        )


def test_evaluates_multiple_cases():
    chunk_x = make_chunk(
        chunk_id="chunk-x",
        ordinal=1,
    )
    chunk_a = make_chunk(
        chunk_id="chunk-a",
        ordinal=2,
    )
    chunk_y = make_chunk(
        chunk_id="chunk-y",
        ordinal=3,
    )

    cases = (
        RetrievalEvaluationCase(
            case_id="case-hit",
            query="query-hit",
            relevant_chunk_ids=[
                "chunk-a",
                "chunk-b",
            ],
        ),
        RetrievalEvaluationCase(
            case_id="case-miss",
            query="query-miss",
            relevant_chunk_ids=[
                "chunk-c",
            ],
        ),
    )
    retriever = FakeRetriever(
        results_by_query={
            "query-hit": (
                chunk_x,
                chunk_a,
            ),
            "query-miss": (
                chunk_y,
            ),
        }
    )

    report = RetrievalEvaluator().evaluate(
        dataset_id="dataset-1",
        cases=cases,
        retriever=retriever,
        top_k=5,
    )

    assert report.case_count == 2
    assert report.hit_rate_at_k == 0.5
    assert report.mean_recall_at_k == 0.25
    assert report.mrr == 0.25
    assert report.retriever == "fake"
    assert report.index_version == (
        "fake-index-v1"
    )


def test_forwards_top_k_and_filters():
    chunk = make_chunk(
        chunk_id="chunk-a",
        ordinal=1,
    )
    filters = SearchFilters(
        page_start=1,
        page_end=1,
    )
    case = RetrievalEvaluationCase(
        case_id="case-1",
        query="query",
        relevant_chunk_ids=[
            "chunk-a",
        ],
        filters=filters,
    )
    retriever = FakeRetriever(
        results_by_query={
            "query": (chunk,),
        }
    )

    RetrievalEvaluator().evaluate(
        dataset_id="dataset",
        cases=(case,),
        retriever=retriever,
        top_k=3,
    )

    assert retriever.calls == [
        (
            "query",
            3,
            filters,
        )
    ]


def test_rejects_empty_cases():
    with pytest.raises(
        ValueError,
        match="cases cannot be empty",
    ):
        RetrievalEvaluator().evaluate(
            dataset_id="dataset",
            cases=(),
            retriever=FakeRetriever(
                results_by_query={}
            ),
        )


def test_rejects_duplicate_case_ids():
    first = RetrievalEvaluationCase(
        case_id="duplicate",
        query="query-1",
        relevant_chunk_ids=["chunk-1"],
    )
    second = RetrievalEvaluationCase(
        case_id="duplicate",
        query="query-2",
        relevant_chunk_ids=["chunk-2"],
    )
    retriever = FakeRetriever(
        results_by_query={}
    )

    with pytest.raises(
        ValueError,
        match=(
            "cases cannot contain "
            "duplicate case IDs"
        ),
    ):
        RetrievalEvaluator().evaluate(
            dataset_id="dataset",
            cases=(first, second),
            retriever=retriever,
        )

    assert retriever.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_rejects_non_positive_top_k(
    top_k: int,
):
    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        RetrievalEvaluator().evaluate(
            dataset_id="dataset",
            cases=(
                RetrievalEvaluationCase(
                    case_id="case-1",
                    query="query",
                    relevant_chunk_ids=[
                        "chunk-1",
                    ],
                ),
            ),
            retriever=FakeRetriever(
                results_by_query={}
            ),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "top_k",
    [True, 1.5],
)
def test_rejects_invalid_top_k_type(
    top_k: object,
):
    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        RetrievalEvaluator().evaluate(
            dataset_id="dataset",
            cases=(
                RetrievalEvaluationCase(
                    case_id="case-1",
                    query="query",
                    relevant_chunk_ids=[
                        "chunk-1",
                    ],
                ),
            ),
            retriever=FakeRetriever(
                results_by_query={}
            ),
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_rejects_changing_index_version():
    chunk = make_chunk(
        chunk_id="chunk-1",
        ordinal=1,
    )
    cases = (
        RetrievalEvaluationCase(
            case_id="case-1",
            query="query-1",
            relevant_chunk_ids=[
                "chunk-1",
            ],
        ),
        RetrievalEvaluationCase(
            case_id="case-2",
            query="query-2",
            relevant_chunk_ids=[
                "chunk-1",
            ],
        ),
    )
    retriever = FakeRetriever(
        results_by_query={
            "query-1": (chunk,),
            "query-2": (chunk,),
        },
        metadata_by_query={
            "query-1": (
                "fake",
                "index-v1",
            ),
            "query-2": (
                "fake",
                "index-v2",
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "index version changed "
            "during evaluation"
        ),
    ):
        RetrievalEvaluator().evaluate(
            dataset_id="dataset",
            cases=cases,
            retriever=retriever,
        )


def test_evaluator_is_exported():
    import rag_lab.evaluation as evaluation

    assert (
        "RetrievalEvaluator"
        in evaluation.__all__
    )
