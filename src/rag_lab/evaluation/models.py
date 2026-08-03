from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from rag_lab.contracts import SearchFilters


class RetrievalEvaluationCase(BaseModel):
    """One labeled query used to evaluate a retriever."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    case_id: str
    query: str
    relevant_chunk_ids: list[str]
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.case_id.strip():
            raise ValueError(
                "case_id cannot be empty"
            )

        if not self.query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not self.relevant_chunk_ids:
            raise ValueError(
                "relevant_chunk_ids cannot be empty"
            )

        if any(
            not chunk_id.strip()
            for chunk_id
            in self.relevant_chunk_ids
        ):
            raise ValueError(
                "relevant_chunk_ids entries "
                "cannot be empty"
            )

        if (
            len(set(self.relevant_chunk_ids))
            != len(self.relevant_chunk_ids)
        ):
            raise ValueError(
                "relevant_chunk_ids cannot "
                "contain duplicates"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class RetrievalCaseResult(BaseModel):
    """Metrics and ranked IDs produced for one evaluation case."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    case_id: str
    query: str

    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]

    top_k: int
    first_relevant_rank: int | None
    hit_at_k: bool
    recall_at_k: float
    reciprocal_rank: float

    @classmethod
    def from_ranked_chunk_ids(
        cls,
        *,
        case: RetrievalEvaluationCase,
        retrieved_chunk_ids: Sequence[str],
        top_k: int,
    ) -> Self:
        ranked_ids = list(
            retrieved_chunk_ids
        )
        relevant_ids = set(
            case.relevant_chunk_ids
        )

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    ranked_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            ),
            None,
        )

        matched_count = len(
            relevant_ids.intersection(
                ranked_ids
            )
        )
        recall_at_k = (
            matched_count
            / len(case.relevant_chunk_ids)
        )
        hit_at_k = first_relevant_rank is not None
        reciprocal_rank = (
            0.0
            if first_relevant_rank is None
            else 1.0 / first_relevant_rank
        )

        return cls(
            case_id=case.case_id,
            query=case.query,
            relevant_chunk_ids=list(
                case.relevant_chunk_ids
            ),
            retrieved_chunk_ids=ranked_ids,
            top_k=top_k,
            first_relevant_rank=(
                first_relevant_rank
            ),
            hit_at_k=hit_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
        )

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.case_id.strip():
            raise ValueError(
                "case_id cannot be empty"
            )

        if not self.query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not self.relevant_chunk_ids:
            raise ValueError(
                "relevant_chunk_ids cannot be empty"
            )

        if any(
            not chunk_id.strip()
            for chunk_id
            in self.relevant_chunk_ids
        ):
            raise ValueError(
                "relevant_chunk_ids entries "
                "cannot be empty"
            )

        if (
            len(set(self.relevant_chunk_ids))
            != len(self.relevant_chunk_ids)
        ):
            raise ValueError(
                "relevant_chunk_ids cannot "
                "contain duplicates"
            )

        if any(
            not chunk_id.strip()
            for chunk_id
            in self.retrieved_chunk_ids
        ):
            raise ValueError(
                "retrieved_chunk_ids entries "
                "cannot be empty"
            )

        if (
            len(set(self.retrieved_chunk_ids))
            != len(self.retrieved_chunk_ids)
        ):
            raise ValueError(
                "retrieved_chunk_ids cannot "
                "contain duplicates"
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        if (
            len(self.retrieved_chunk_ids)
            > self.top_k
        ):
            raise ValueError(
                "retrieved_chunk_ids cannot "
                "exceed top_k"
            )

        for field_name, value in {
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": (
                self.reciprocal_rank
            ),
        }.items():
            if (
                not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"{field_name} must be finite "
                    "and between 0 and 1"
                )

        relevant_ids = set(
            self.relevant_chunk_ids
        )

        expected_first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    self.retrieved_chunk_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            ),
            None,
        )

        if (
            self.first_relevant_rank
            != expected_first_rank
        ):
            raise ValueError(
                "first_relevant_rank does not "
                "match retrieved order"
            )

        expected_hit = (
            expected_first_rank is not None
        )

        if self.hit_at_k != expected_hit:
            raise ValueError(
                "hit_at_k does not match "
                "retrieved IDs"
            )

        expected_recall = (
            len(
                relevant_ids.intersection(
                    self.retrieved_chunk_ids
                )
            )
            / len(self.relevant_chunk_ids)
        )

        if not math.isclose(
            self.recall_at_k,
            expected_recall,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "recall_at_k does not match "
                "retrieved IDs"
            )

        expected_reciprocal_rank = (
            0.0
            if expected_first_rank is None
            else 1.0 / expected_first_rank
        )

        if not math.isclose(
            self.reciprocal_rank,
            expected_reciprocal_rank,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "reciprocal_rank does not match "
                "first relevant rank"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

class RetrievalEvaluationReport(BaseModel):
    """Aggregate metrics from one retrieval evaluation run."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    dataset_id: str
    retriever: str
    index_version: str
    top_k: int

    case_results: list[RetrievalCaseResult]
    case_count: int

    hit_rate_at_k: float
    mean_recall_at_k: float
    mrr: float

    @classmethod
    def from_case_results(
        cls,
        *,
        dataset_id: str,
        retriever: str,
        index_version: str,
        top_k: int,
        case_results: Sequence[
            RetrievalCaseResult
        ],
    ) -> Self:
        results = list(case_results)
        case_count = len(results)

        denominator = (
            case_count
            if case_count
            else 1
        )

        hit_rate_at_k = (
            sum(
                result.hit_at_k
                for result in results
            )
            / denominator
        )
        mean_recall_at_k = (
            sum(
                result.recall_at_k
                for result in results
            )
            / denominator
        )
        mrr = (
            sum(
                result.reciprocal_rank
                for result in results
            )
            / denominator
        )

        return cls(
            dataset_id=dataset_id,
            retriever=retriever,
            index_version=index_version,
            top_k=top_k,
            case_results=results,
            case_count=case_count,
            hit_rate_at_k=hit_rate_at_k,
            mean_recall_at_k=mean_recall_at_k,
            mrr=mrr,
        )

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        required_strings = {
            "dataset_id": self.dataset_id,
            "retriever": self.retriever,
            "index_version": self.index_version,
        }

        for field_name, value in (
            required_strings.items()
        ):
            if not value.strip():
                raise ValueError(
                    f"{field_name} cannot be empty"
                )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        if not self.case_results:
            raise ValueError(
                "case_results cannot be empty"
            )

        if self.case_count != len(
            self.case_results
        ):
            raise ValueError(
                "case_count does not match "
                "case_results"
            )

        case_ids = [
            result.case_id
            for result in self.case_results
        ]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "case_results cannot contain "
                "duplicate case IDs"
            )

        if any(
            result.top_k != self.top_k
            for result in self.case_results
        ):
            raise ValueError(
                "case result top_k does not "
                "match report top_k"
            )

        metrics = {
            "hit_rate_at_k": self.hit_rate_at_k,
            "mean_recall_at_k": (
                self.mean_recall_at_k
            ),
            "mrr": self.mrr,
        }

        for field_name, value in metrics.items():
            if (
                not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"{field_name} must be finite "
                    "and between 0 and 1"
                )

        denominator = len(self.case_results)

        expected_metrics = {
            "hit_rate_at_k": (
                sum(
                    result.hit_at_k
                    for result
                    in self.case_results
                )
                / denominator
            ),
            "mean_recall_at_k": (
                sum(
                    result.recall_at_k
                    for result
                    in self.case_results
                )
                / denominator
            ),
            "mrr": (
                sum(
                    result.reciprocal_rank
                    for result
                    in self.case_results
                )
                / denominator
            ),
        }

        for field_name, expected_value in (
            expected_metrics.items()
        ):
            actual_value = getattr(
                self,
                field_name,
            )

            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{field_name} does not "
                    "match case_results"
                )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
