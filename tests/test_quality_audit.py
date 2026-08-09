from __future__ import annotations

import json
from pathlib import Path

from rag_lab.quality import (
    ArtifactQualityInputs,
    AuditConfig,
    audit_artifacts,
)
from rag_lab.quality.cli import main


def make_block(
    *,
    block_id: str,
    text: str,
    ordinal: int,
    page_start: int = 1,
    page_end: int = 1,
    block_type: str = "paragraph",
    image_path: str | None = None,
) -> dict[str, object]:
    return {
        "block_id": block_id,
        "document_id": "document-a",
        "text": text,
        "block_type": block_type,
        "heading_path": ["Chapter one"],
        "page_start": page_start,
        "page_end": page_end,
        "ordinal": ordinal,
        "source_path": "book.pdf",
        "image_path": image_path,
        "normalization_version": "v1",
    }


def make_chunk(
    *,
    chunk_id: str,
    block_ids: list[object],
    content: str = "content",
    index_text: str = "Chapter one\n\ncontent",
    ordinal: int = 1,
    page_start: int = 1,
    page_end: int = 1,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "document-a",
        "content": content,
        "index_text": index_text,
        "heading_path": ["Chapter one"],
        "page_start": page_start,
        "page_end": page_end,
        "ordinal": ordinal,
        "block_ids": block_ids,
        "source_path": "book.pdf",
        "content_hash": "hash",
        "normalization_version": "v1",
        "chunking_version": "v1",
    }


def write_bundle(
    tmp_path: Path,
    *,
    markdown: str = "source text\n",
    blocks: list[dict[str, object]] | None = None,
    chunks: list[dict[str, object]] | None = None,
    normalization_report: dict[str, object] | None = None,
    chunking_report: dict[str, object] | None = None,
) -> ArtifactQualityInputs:
    root = tmp_path / "artifact"
    normalized = root / "normalized"
    chunked = root / "chunked-max1200-overlap120"
    normalized.mkdir(parents=True)
    chunked.mkdir()
    (root / "source.md").write_text(
        markdown,
        encoding="utf-8",
    )

    blocks = blocks or [
        make_block(block_id="block-a", text="正文。", ordinal=1)
    ]
    chunks = chunks or [
        make_chunk(chunk_id="chunk-a", block_ids=["block-a"])
    ]
    normalization_report = normalization_report or {
        "document_id": "document-a",
        "normalized_block_count": len(blocks),
        "short_fragment_ratio": 0.0,
        "source_pages": [1],
        "pages_requiring_review": [],
    }
    chunking_report = chunking_report or {
        "input_block_count": len(blocks),
        "output_chunk_count": len(chunks),
    }

    (normalized / "blocks.jsonl").write_text(
        "\n".join(
            json.dumps(block, ensure_ascii=False)
            for block in blocks
        )
        + "\n",
        encoding="utf-8",
    )
    (chunked / "chunks.jsonl").write_text(
        "\n".join(
            json.dumps(chunk, ensure_ascii=False)
            for chunk in chunks
        )
        + "\n",
        encoding="utf-8",
    )
    (normalized / "normalization-report.json").write_text(
        json.dumps(normalization_report),
        encoding="utf-8",
    )
    (chunked / "chunking-report.json").write_text(
        json.dumps(chunking_report),
        encoding="utf-8",
    )

    return ArtifactQualityInputs(
        docling_markdown=root / "source.md",
        normalization_report=(
            normalized / "normalization-report.json"
        ),
        blocks=normalized / "blocks.jsonl",
        chunking_report=chunked / "chunking-report.json",
        chunks=chunked / "chunks.jsonl",
    )


def issue_codes(report) -> set[str]:
    return {
        issue.issue_code
        for issue in report.issues
    }


def test_audit_detects_structural_errors_and_formula_markers(
    tmp_path: Path,
):
    blocks = [
        make_block(
            block_id="block-empty",
            text="",
            ordinal=1,
            page_start=0,
            page_end=0,
        ),
        make_block(
            block_id="block-punctuation",
            text="。",
            ordinal=2,
        ),
        make_block(
            block_id="block-image",
            text="图注",
            ordinal=3,
            image_path="assets/missing.png",
        ),
    ]
    chunks = [
        make_chunk(
            chunk_id="chunk-empty",
            block_ids=["block-punctuation", "missing", "missing"],
            content="",
            index_text="X" * 20,
            page_start=2,
            page_end=1,
        )
    ]
    inputs = write_bundle(
        tmp_path,
        markdown=(
            "before\n<!-- formula-not-decoded -->\nafter\n"
        ),
        blocks=blocks,
        chunks=chunks,
        normalization_report={
            "document_id": "document-a",
            "normalized_block_count": 99,
            "short_fragment_ratio": 0.0,
            "source_pages": [1],
            "pages_requiring_review": [],
        },
        chunking_report={
            "input_block_count": 99,
            "output_chunk_count": 99,
        },
    )

    report = audit_artifacts(
        inputs=inputs,
        config=AuditConfig(max_chunk_chars=5),
    )

    assert report.passed is False
    assert {
        "FORMULA_NOT_DECODED",
        "EMPTY_BLOCK",
        "EMPTY_CHUNK",
        "ORPHAN_PUNCTUATION_BLOCK",
        "MISSING_IMAGE_TARGET",
        "REPORT_COUNT_MISMATCH",
        "BROKEN_BLOCK_REFERENCE",
        "INVALID_PAGE_RANGE",
        "CHUNK_EXCEEDS_CONFIGURED_LIMIT",
    } <= issue_codes(report)


def test_audit_marks_source_formula_as_restored_when_overlay_reports_it(
    tmp_path: Path,
):
    inputs = write_bundle(
        tmp_path,
        markdown="<!-- formula-not-decoded -->\n",
        blocks=[
            make_block(
                block_id="equation-a",
                text="d = L / R",
                ordinal=1,
                block_type="equation",
            )
        ],
        chunks=[make_chunk(chunk_id="chunk-a", block_ids=["equation-a"])],
        normalization_report={
            "document_id": "document-a",
            "normalized_block_count": 1,
            "short_fragment_ratio": 0.0,
            "source_pages": [1],
            "pages_requiring_review": [],
            "correction_overlay": {
                "formula_restorations": [
                    {
                        "correction_id": "restore-equation",
                        "marker_line": 1,
                        "marker_page": 1,
                        "source_ref": "#/texts/4",
                        "equation_block_id": "equation-a",
                    }
                ]
            },
        },
    )

    report = audit_artifacts(inputs=inputs)

    assert report.passed is True
    restored = [
        issue
        for issue in report.issues
        if issue.issue_code == "FORMULA_NOT_DECODED"
    ]
    assert restored[0].severity == "warning"
    assert "FORMULA_RESTORED_BY_CORRECTION" in issue_codes(report)


def test_audit_warns_conservatively_for_review_patterns(
    tmp_path: Path,
):
    blocks = [
        make_block(block_id="block-text", text="unfinished", ordinal=1),
        make_block(block_id="block-dot", text="。", ordinal=2),
        make_block(block_id="block-la", text="La", ordinal=3),
        make_block(block_id="block-slash", text="/", ordinal=4),
        make_block(block_id="block-r", text="R", ordinal=5),
        make_block(
            block_id="block-caption",
            text="图 1-18",
            ordinal=6,
            block_type="figure_caption",
        ),
        make_block(block_id="block-heading", text="1. T", ordinal=7),
        make_block(
            block_id="block-spacing",
            text="见 www. example. com。",
            ordinal=8,
        ),
        make_block(
            block_id="block-semantic",
            text="应用层",
            ordinal=9,
        ),
        make_block(
            block_id="block-short",
            text="a",
            ordinal=10,
        ),
    ]
    inputs = write_bundle(
        tmp_path,
        blocks=blocks,
        chunks=[
            make_chunk(
                chunk_id="chunk-a",
                block_ids=[
                    str(block["block_id"])
                    for block in blocks
                ],
            )
        ],
        normalization_report={
            "document_id": "document-a",
            "normalized_block_count": len(blocks),
            "short_fragment_ratio": 0.05,
            "source_pages": [1, 2, 3, 4],
            "pages_requiring_review": [1, 2],
        },
    )

    report = audit_artifacts(inputs=inputs)

    assert {
        "VERY_SHORT_TEXT_BLOCK",
        "POSSIBLE_FIGURE_LABEL_SEQUENCE",
        "POSSIBLE_READING_ORDER_BREAK",
        "HEADING_LIKE_PARAGRAPH",
        "SUSPICIOUS_WORD_SPACING",
        "HIGH_SHORT_FRAGMENT_RATIO",
        "EXCESSIVE_REVIEW_PAGE_RATIO",
    } <= issue_codes(report)
    short_text_warnings = [
        issue
        for issue in report.issues
        if issue.issue_code == "VERY_SHORT_TEXT_BLOCK"
    ]
    assert "block-short" in [
        issue.block_id
        for issue in short_text_warnings
    ]
    assert "block-semantic" not in [
        issue.block_id
        for issue in short_text_warnings
    ]


def test_cli_audits_artifact_root_and_writes_report(
    tmp_path: Path,
    capsys,
):
    inputs = write_bundle(tmp_path)
    root = inputs.docling_markdown.parent

    exit_code = main(["--artifact-root", str(root)])

    captured = capsys.readouterr()
    report_path = root / "artifact-quality-report.json"
    assert exit_code == 0
    assert "Status: passed" in captured.out
    assert report_path.is_file()
    assert json.loads(
        report_path.read_text(encoding="utf-8")
    )["passed"] is True
