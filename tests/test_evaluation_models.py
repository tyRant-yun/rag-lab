import pytest
from pydantic import ValidationError

import rag_lab.evaluation as evaluation
from rag_lab.contracts import SearchFilters
from rag_lab.evaluation import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
)


def make_case() -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id="protocol-definition",
        query="什么是协议",
        relevant_chunk_ids=[
            "chunk-protocol",
        ],
    )


def test_constructs_evaluation_case():
    case = make_case()

    assert case.case_id == (
        "protocol-definition"
    )
    assert case.query == "什么是协议"
    assert case.relevant_chunk_ids == [
        "chunk-protocol"
    ]
    assert case.filters is None


def test_serializes_nested_filters():
    case = RetrievalEvaluationCase(
        case_id="protocol-pages",
        query="什么是协议",
        relevant_chunk_ids=[
            "chunk-protocol",
        ],
        filters=SearchFilters(
            page_start=23,
            page_end=23,
        ),
    )

    assert case.to_dict() == {
        "case_id": "protocol-pages",
        "query": "什么是协议",
        "relevant_chunk_ids": [
            "chunk-protocol",
        ],
        "filters": {
            "document_ids": None,
            "heading_prefix": None,
            "page_start": 23,
            "page_end": 23,
        },
    }


@pytest.mark.parametrize(
    "field_name",
    [
        "case_id",
        "query",
    ],
)
def test_rejects_empty_required_string(
    field_name: str,
):
    values = {
        "case_id": "case-1",
        "query": "协议",
        "relevant_chunk_ids": [
            "chunk-1",
        ],
    }
    values[field_name] = " "

    with pytest.raises(
        ValidationError,
        match=f"{field_name} cannot be empty",
    ):
        RetrievalEvaluationCase(**values)


def test_rejects_empty_relevant_chunk_ids():
    with pytest.raises(
        ValidationError,
        match=(
            "relevant_chunk_ids cannot be empty"
        ),
    ):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[],
        )


def test_rejects_blank_relevant_chunk_id():
    with pytest.raises(
        ValidationError,
        match=(
            "relevant_chunk_ids entries "
            "cannot be empty"
        ),
    ):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-1",
                " ",
            ],
        )


def test_rejects_duplicate_relevant_chunk_ids():
    with pytest.raises(
        ValidationError,
        match=(
            "relevant_chunk_ids cannot "
            "contain duplicates"
        ),
    ):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-1",
                "chunk-1",
            ],
        )


def test_is_frozen():
    case = make_case()

    with pytest.raises(ValidationError):
        case.query = "另一个查询"


def test_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-1",
            ],
            unknown="value",
        )


def test_requires_strict_field_types():
    with pytest.raises(ValidationError):
        RetrievalEvaluationCase(
            case_id="case-1",
            query=123,
            relevant_chunk_ids=[
                "chunk-1",
            ],
        )


def test_is_exported():
    assert (
        "RetrievalEvaluationCase"
        in evaluation.__all__
    )

def test_builds_case_result_from_ranked_ids():
    case = RetrievalEvaluationCase(
        case_id="protocol-definition",
        query="什么是协议",
        relevant_chunk_ids=[
            "chunk-protocol",
            "chunk-standard",
        ],
    )

    result = (
        RetrievalCaseResult
        .from_ranked_chunk_ids(
            case=case,
            retrieved_chunk_ids=[
                "chunk-unrelated",
                "chunk-protocol",
            ],
            top_k=5,
        )
    )

    assert result.first_relevant_rank == 2
    assert result.hit_at_k is True
    assert result.recall_at_k == 0.5
    assert result.reciprocal_rank == 0.5


def test_builds_no_hit_case_result():
    case = make_case()

    result = (
        RetrievalCaseResult
        .from_ranked_chunk_ids(
            case=case,
            retrieved_chunk_ids=[
                "chunk-unrelated",
            ],
            top_k=5,
        )
    )

    assert result.first_relevant_rank is None
    assert result.hit_at_k is False
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0


def test_rejects_too_many_retrieved_ids():
    case = make_case()

    with pytest.raises(
        ValidationError,
        match=(
            "retrieved_chunk_ids cannot "
            "exceed top_k"
        ),
    ):
        (
            RetrievalCaseResult
            .from_ranked_chunk_ids(
                case=case,
                retrieved_chunk_ids=[
                    "chunk-1",
                    "chunk-2",
                ],
                top_k=1,
            )
        )


def test_rejects_duplicate_retrieved_ids():
    case = make_case()

    with pytest.raises(
        ValidationError,
        match=(
            "retrieved_chunk_ids cannot "
            "contain duplicates"
        ),
    ):
        (
            RetrievalCaseResult
            .from_ranked_chunk_ids(
                case=case,
                retrieved_chunk_ids=[
                    "chunk-1",
                    "chunk-1",
                ],
                top_k=5,
            )
        )


def test_rejects_inconsistent_first_rank():
    with pytest.raises(
        ValidationError,
        match=(
            "first_relevant_rank does not "
            "match retrieved order"
        ),
    ):
        RetrievalCaseResult(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-relevant",
            ],
            retrieved_chunk_ids=[
                "chunk-relevant",
            ],
            top_k=5,
            first_relevant_rank=None,
            hit_at_k=True,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
        )


def test_rejects_inconsistent_hit():
    with pytest.raises(
        ValidationError,
        match=(
            "hit_at_k does not match "
            "retrieved IDs"
        ),
    ):
        RetrievalCaseResult(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-relevant",
            ],
            retrieved_chunk_ids=[
                "chunk-relevant",
            ],
            top_k=5,
            first_relevant_rank=1,
            hit_at_k=False,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
        )


def test_rejects_inconsistent_recall():
    with pytest.raises(
        ValidationError,
        match=(
            "recall_at_k does not match "
            "retrieved IDs"
        ),
    ):
        RetrievalCaseResult(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-relevant",
            ],
            retrieved_chunk_ids=[
                "chunk-relevant",
            ],
            top_k=5,
            first_relevant_rank=1,
            hit_at_k=True,
            recall_at_k=0.5,
            reciprocal_rank=1.0,
        )


def test_rejects_inconsistent_reciprocal_rank():
    with pytest.raises(
        ValidationError,
        match=(
            "reciprocal_rank does not match "
            "first relevant rank"
        ),
    ):
        RetrievalCaseResult(
            case_id="case-1",
            query="协议",
            relevant_chunk_ids=[
                "chunk-relevant",
            ],
            retrieved_chunk_ids=[
                "chunk-unrelated",
                "chunk-relevant",
            ],
            top_k=5,
            first_relevant_rank=2,
            hit_at_k=True,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
        )


def test_case_result_is_exported():
    assert (
        "RetrievalCaseResult"
        in evaluation.__all__
    )

def make_case_result(
    *,
    case_id: str,
    relevant_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    top_k: int = 5,
) -> RetrievalCaseResult:
    case = RetrievalEvaluationCase(
        case_id=case_id,
        query=f"query-{case_id}",
        relevant_chunk_ids=(
            relevant_chunk_ids
        ),
    )

    return (
        RetrievalCaseResult
        .from_ranked_chunk_ids(
            case=case,
            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),
            top_k=top_k,
        )
    )


def make_report_results(
) -> tuple[RetrievalCaseResult, ...]:
    return (
        make_case_result(
            case_id="case-hit",
            relevant_chunk_ids=[
                "chunk-a",
                "chunk-b",
            ],
            retrieved_chunk_ids=[
                "chunk-x",
                "chunk-a",
            ],
        ),
        make_case_result(
            case_id="case-miss",
            relevant_chunk_ids=[
                "chunk-c",
            ],
            retrieved_chunk_ids=[
                "chunk-y",
            ],
        ),
    )


def test_builds_aggregate_report():
    report = (
        RetrievalEvaluationReport
        .from_case_results(
            dataset_id="chapter-01-smoke",
            retriever="bm25",
            index_version="bm25-v1:test",
            top_k=5,
            case_results=make_report_results(),
        )
    )

    assert report.case_count == 2
    assert report.hit_rate_at_k == 0.5
    assert report.mean_recall_at_k == 0.25
    assert report.mrr == 0.25


def test_serializes_aggregate_report():
    report = (
        RetrievalEvaluationReport
        .from_case_results(
            dataset_id="chapter-01-smoke",
            retriever="bm25",
            index_version="bm25-v1:test",
            top_k=5,
            case_results=make_report_results(),
        )
    )

    payload = report.to_dict()

    assert payload["dataset_id"] == (
        "chapter-01-smoke"
    )
    assert payload["retriever"] == "bm25"
    assert payload["case_count"] == 2
    assert len(payload["case_results"]) == 2


def test_rejects_empty_case_results():
    with pytest.raises(
        ValidationError,
        match="case_results cannot be empty",
    ):
        (
            RetrievalEvaluationReport
            .from_case_results(
                dataset_id="dataset",
                retriever="bm25",
                index_version="bm25-v1:test",
                top_k=5,
                case_results=(),
            )
        )


def test_rejects_duplicate_case_ids():
    result = make_case_result(
        case_id="duplicate",
        relevant_chunk_ids=[
            "chunk-a",
        ],
        retrieved_chunk_ids=[
            "chunk-a",
        ],
    )

    with pytest.raises(
        ValidationError,
        match=(
            "case_results cannot contain "
            "duplicate case IDs"
        ),
    ):
        (
            RetrievalEvaluationReport
            .from_case_results(
                dataset_id="dataset",
                retriever="bm25",
                index_version="bm25-v1:test",
                top_k=5,
                case_results=(
                    result,
                    result,
                ),
            )
        )


def test_rejects_mismatched_result_top_k():
    result = make_case_result(
        case_id="case-1",
        relevant_chunk_ids=[
            "chunk-a",
        ],
        retrieved_chunk_ids=[
            "chunk-a",
        ],
        top_k=3,
    )

    with pytest.raises(
        ValidationError,
        match=(
            "case result top_k does not "
            "match report top_k"
        ),
    ):
        (
            RetrievalEvaluationReport
            .from_case_results(
                dataset_id="dataset",
                retriever="bm25",
                index_version="bm25-v1:test",
                top_k=5,
                case_results=(result,),
            )
        )


def test_rejects_incorrect_case_count():
    results = list(make_report_results())

    with pytest.raises(
        ValidationError,
        match=(
            "case_count does not match "
            "case_results"
        ),
    ):
        RetrievalEvaluationReport(
            dataset_id="dataset",
            retriever="bm25",
            index_version="bm25-v1:test",
            top_k=5,
            case_results=results,
            case_count=3,
            hit_rate_at_k=0.5,
            mean_recall_at_k=0.25,
            mrr=0.25,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "hit_rate_at_k",
        "mean_recall_at_k",
        "mrr",
    ],
)
def test_rejects_inconsistent_report_metric(
    field_name: str,
):
    values = {
        "dataset_id": "dataset",
        "retriever": "bm25",
        "index_version": "bm25-v1:test",
        "top_k": 5,
        "case_results": list(
            make_report_results()
        ),
        "case_count": 2,
        "hit_rate_at_k": 0.5,
        "mean_recall_at_k": 0.25,
        "mrr": 0.25,
    }
    values[field_name] = 0.75

    with pytest.raises(
        ValidationError,
        match=(
            f"{field_name} does not "
            "match case_results"
        ),
    ):
        RetrievalEvaluationReport(**values)


def test_evaluation_report_is_exported():
    assert (
        "RetrievalEvaluationReport"
        in evaluation.__all__
    )
