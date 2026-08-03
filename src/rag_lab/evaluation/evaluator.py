from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.evaluation.models import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
)


class RetrievalSearcher(Protocol):
    """Minimal retriever interface required by evaluation."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        ...


class RetrievalEvaluator:
    """Evaluate any retriever that returns SearchResult."""

    def evaluate(
        self,
        *,
        dataset_id: str,
        cases: Sequence[
            RetrievalEvaluationCase
        ],
        retriever: RetrievalSearcher,
        top_k: int = 5,
    ) -> RetrievalEvaluationReport:
        if not isinstance(dataset_id, str):
            raise TypeError(
                "dataset_id must be a string"
            )

        if not dataset_id.strip():
            raise ValueError(
                "dataset_id cannot be empty"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        resolved_cases = tuple(cases)

        if not resolved_cases:
            raise ValueError(
                "cases cannot be empty"
            )

        case_ids = [
            case.case_id
            for case in resolved_cases
        ]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "cases cannot contain "
                "duplicate case IDs"
            )

        case_results: list[
            RetrievalCaseResult
        ] = []

        report_retriever: str | None = None
        report_index_version: str | None = None

        for case in resolved_cases:
            search_result = retriever.search(
                case.query,
                top_k=top_k,
                filters=case.filters,
            )

            if not isinstance(
                search_result,
                SearchResult,
            ):
                raise TypeError(
                    "retriever.search must return "
                    "SearchResult"
                )

            if search_result.query != case.query:
                raise ValueError(
                    f"case {case.case_id!r}: "
                    "search result query does not "
                    "match evaluation query"
                )

            if report_retriever is None:
                report_retriever = (
                    search_result.retriever
                )
                report_index_version = (
                    search_result.index_version
                )
            else:
                if (
                    search_result.retriever
                    != report_retriever
                ):
                    raise ValueError(
                        "retriever name changed "
                        "during evaluation"
                    )

                if (
                    search_result.index_version
                    != report_index_version
                ):
                    raise ValueError(
                        "index version changed "
                        "during evaluation"
                    )

            retrieved_chunk_ids = [
                hit.chunk.chunk_id
                for hit in search_result.hits
            ]

            case_results.append(
                RetrievalCaseResult
                .from_ranked_chunk_ids(
                    case=case,
                    retrieved_chunk_ids=(
                        retrieved_chunk_ids
                    ),
                    top_k=top_k,
                )
            )

        if (
            report_retriever is None
            or report_index_version is None
        ):
            raise RuntimeError(
                "evaluation produced no results"
            )

        return (
            RetrievalEvaluationReport
            .from_case_results(
                dataset_id=dataset_id,
                retriever=report_retriever,
                index_version=(
                    report_index_version
                ),
                top_k=top_k,
                case_results=case_results,
            )
        )
