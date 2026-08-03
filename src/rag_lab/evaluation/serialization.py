from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from pydantic import ValidationError

from rag_lab.evaluation.models import (
    RetrievalEvaluationCase,
)


def read_retrieval_evaluation_cases_jsonl(
    path: Path,
    *,
    known_chunk_ids: Collection[str]
    | None = None,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Read and validate retrieval evaluation cases."""

    cases: list[RetrievalEvaluationCase] = []
    seen_case_ids: set[str] = set()

    resolved_chunk_ids = (
        None
        if known_chunk_ids is None
        else frozenset(known_chunk_ids)
    )

    with path.open(
        encoding="utf-8",
    ) as stream:
        for line_number, raw_line in enumerate(
            stream,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "empty JSONL record"
                )

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "invalid JSON"
                ) from error

            try:
                case = (
                    RetrievalEvaluationCase
                    .model_validate(
                        payload,
                        strict=True,
                    )
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "invalid RetrievalEvaluationCase"
                ) from error

            if case.case_id in seen_case_ids:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    f"duplicate case_id "
                    f"{case.case_id!r}"
                )

            if resolved_chunk_ids is not None:
                unknown_chunk_ids = [
                    chunk_id
                    for chunk_id
                    in case.relevant_chunk_ids
                    if chunk_id
                    not in resolved_chunk_ids
                ]

                if unknown_chunk_ids:
                    rendered_ids = ", ".join(
                        repr(chunk_id)
                        for chunk_id
                        in unknown_chunk_ids
                    )

                    raise ValueError(
                        f"{path}: line {line_number}: "
                        "unknown relevant chunk IDs: "
                        f"{rendered_ids}"
                    )

            seen_case_ids.add(case.case_id)
            cases.append(case)

    return tuple(cases)
