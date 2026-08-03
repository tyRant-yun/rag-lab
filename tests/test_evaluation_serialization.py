from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lab.contracts import SearchFilters
from rag_lab.evaluation import (
    RetrievalEvaluationCase,
    read_retrieval_evaluation_cases_jsonl,
)


def make_case(
    *,
    case_id: str,
    relevant_chunk_ids: list[str],
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        query=f"query-{case_id}",
        relevant_chunk_ids=(
            relevant_chunk_ids
        ),
    )


def write_jsonl(
    path: Path,
    payloads: list[dict[str, object]],
) -> None:
    content = "\n".join(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
        for payload in payloads
    )

    if payloads:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def test_reads_cases_in_file_order(
    tmp_path: Path,
):
    first = make_case(
        case_id="case-1",
        relevant_chunk_ids=["chunk-1"],
    )
    second = RetrievalEvaluationCase(
        case_id="case-2",
        query="协议是什么",
        relevant_chunk_ids=["chunk-2"],
        filters=SearchFilters(
            page_start=23,
            page_end=23,
        ),
    )
    path = tmp_path / "cases.jsonl"

    write_jsonl(
        path,
        [
            first.to_dict(),
            second.to_dict(),
        ],
    )

    cases = (
        read_retrieval_evaluation_cases_jsonl(
            path
        )
    )

    assert cases == (first, second)


def test_empty_file_returns_empty_tuple(
    tmp_path: Path,
):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    assert (
        read_retrieval_evaluation_cases_jsonl(
            path
        )
        == ()
    )


def test_rejects_empty_record(
    tmp_path: Path,
):
    case = make_case(
        case_id="case-1",
        relevant_chunk_ids=["chunk-1"],
    )
    path = tmp_path / "cases.jsonl"

    path.write_text(
        json.dumps(
            case.to_dict(),
            ensure_ascii=False,
        )
        + "\n\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        ValueError,
        match=r"line 2: empty JSONL record",
    ):
        read_retrieval_evaluation_cases_jsonl(
            path
        )


def test_reports_invalid_json_line(
    tmp_path: Path,
):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":\n',
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        ValueError,
        match=r"line 1: invalid JSON",
    ):
        read_retrieval_evaluation_cases_jsonl(
            path
        )


def test_reports_invalid_case_line(
    tmp_path: Path,
):
    case = make_case(
        case_id="case-1",
        relevant_chunk_ids=["chunk-1"],
    )
    payload = case.to_dict()
    payload["query"] = ""

    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [payload])

    with pytest.raises(
        ValueError,
        match=(
            r"line 1: invalid "
            r"RetrievalEvaluationCase"
        ),
    ):
        read_retrieval_evaluation_cases_jsonl(
            path
        )


def test_rejects_duplicate_case_ids(
    tmp_path: Path,
):
    first = make_case(
        case_id="duplicate",
        relevant_chunk_ids=["chunk-1"],
    )
    second = make_case(
        case_id="duplicate",
        relevant_chunk_ids=["chunk-2"],
    )
    path = tmp_path / "cases.jsonl"

    write_jsonl(
        path,
        [
            first.to_dict(),
            second.to_dict(),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"line 2: duplicate case_id "
            r"'duplicate'"
        ),
    ):
        read_retrieval_evaluation_cases_jsonl(
            path
        )


def test_accepts_known_relevant_chunk_ids(
    tmp_path: Path,
):
    case = make_case(
        case_id="case-1",
        relevant_chunk_ids=[
            "chunk-1",
            "chunk-2",
        ],
    )
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [case.to_dict()])

    cases = (
        read_retrieval_evaluation_cases_jsonl(
            path,
            known_chunk_ids={
                "chunk-1",
                "chunk-2",
                "chunk-3",
            },
        )
    )

    assert cases == (case,)


def test_rejects_unknown_relevant_chunk_id(
    tmp_path: Path,
):
    case = make_case(
        case_id="case-1",
        relevant_chunk_ids=[
            "chunk-known",
            "chunk-missing",
        ],
    )
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [case.to_dict()])

    with pytest.raises(
        ValueError,
        match=(
            r"line 1: unknown relevant "
            r"chunk IDs: 'chunk-missing'"
        ),
    ):
        read_retrieval_evaluation_cases_jsonl(
            path,
            known_chunk_ids={
                "chunk-known",
            },
        )
