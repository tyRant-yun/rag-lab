from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lab.chunking.cli import main
from rag_lab.chunking.serialization import (
    read_normalized_blocks_jsonl,
)


def build_record(
    *,
    ordinal: int = 1,
    text: str = "测试正文",
) -> dict[str, object]:
    return {
        "block_id": f"sha256:block-{ordinal}",
        "document_id": "sha256:document",
        "text": text,
        "block_type": "paragraph",
        "heading_path": ["第1章"],
        "page_start": 19,
        "page_end": 19,
        "ordinal": ordinal,
        "source_path": "D:/source.pdf",
        "image_path": None,
        "normalization_version": "1.0.0",
    }


def write_records(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    content = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    )

    path.write_text(
        content + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_reader_parses_normalized_blocks(
    tmp_path: Path,
):
    input_path = tmp_path / "blocks.jsonl"

    write_records(
        input_path,
        [
            build_record(ordinal=1),
            build_record(ordinal=2),
        ],
    )

    blocks = read_normalized_blocks_jsonl(
        input_path
    )

    assert len(blocks) == 2
    assert blocks[0].ordinal == 1
    assert blocks[1].ordinal == 2


def test_cli_writes_chunking_outputs(
    tmp_path: Path,
    capsys,
):
    input_path = tmp_path / "blocks.jsonl"
    output_path = tmp_path / "output"

    write_records(
        input_path,
        [
            build_record(
                ordinal=1,
                text="A" * 60,
            ),
            build_record(
                ordinal=2,
                text="B" * 60,
            ),
        ],
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--max-chars",
            "100",
            "--chunking-version",
            "1.2.0",
        ]
    )

    assert exit_code == 0

    records = [
        json.loads(line)
        for line in (
            output_path / "chunks.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 2
    assert all(
        record["chunking_version"] == "1.2.0"
        for record in records
    )

    assert (
        output_path / "chunks.md"
    ).is_file()
    assert (
        output_path
        / "chunking-report.json"
    ).is_file()

    captured = capsys.readouterr()

    assert (
        "chunked 2 blocks into 2 chunks"
        in captured.out
    )


def test_reader_reports_invalid_json_line(
    tmp_path: Path,
):
    input_path = tmp_path / "blocks.jsonl"
    input_path.write_text(
        '{"invalid":\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"line 1: invalid JSON",
    ):
        read_normalized_blocks_jsonl(
            input_path
        )


def test_reader_reports_invalid_contract_line(
    tmp_path: Path,
):
    input_path = tmp_path / "blocks.jsonl"
    input_path.write_text(
        '{"text":"missing fields"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"line 1: invalid "
            r"NormalizedBlock"
        ),
    ):
        read_normalized_blocks_jsonl(
            input_path
        )


def test_cli_rejects_empty_jsonl_record(
    tmp_path: Path,
    capsys,
):
    input_path = tmp_path / "blocks.jsonl"
    output_path = tmp_path / "output"

    input_path.write_text(
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
    ) as error:
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )

    assert error.value.code == 2

    captured = capsys.readouterr()

    assert "empty JSONL record" in captured.err
