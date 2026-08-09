from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_lab.quality.models import (
    ArtifactQualityInputs,
    ArtifactQualityIssue,
    ArtifactQualityReport,
    AuditConfig,
    IssueSeverity,
)


_ORPHAN_PUNCTUATION = frozenset(
    {
        "。",
        "，",
        "；",
        "：",
        "、",
        ".",
        ",",
        ";",
        ":",
    }
)
_SENTENCE_TERMINATORS = frozenset(
    {"。", "！", "？", ".", "!", "?"}
)
_FIGURE_LABEL = re.compile(
    r"^(?:[A-Za-z]{1,3}|/|[）)]|[（(]?[a-zA-Z0-9][）)]?)$"
)
_NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.、]?)\s+\S+"
)
_SUSPICIOUS_SPACING = re.compile(
    r"(?:https?://|www\.|\b\w+\.)\s+[A-Za-z0-9]"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {path}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL in {path} line {line_number}: {error}"
            ) from error

        if not isinstance(value, dict):
            raise ValueError(
                f"JSON object expected in {path} "
                f"line {line_number}"
            )

        rows.append(value)

    return rows


def _string(value: object) -> str | None:
    if isinstance(value, str):
        return value

    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _document_id(
    normalization_report: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> str | None:
    report_id = _string(normalization_report.get("document_id"))
    if report_id:
        return report_id

    if blocks:
        return _string(blocks[0].get("document_id"))

    return None


def _issue(
    *,
    issue_code: str,
    severity: IssueSeverity,
    pipeline_stage: str,
    document_id: str | None,
    evidence: str,
    remediation_hint: str,
    page: int | None = None,
    block: dict[str, Any] | None = None,
    chunk: dict[str, Any] | None = None,
) -> ArtifactQualityIssue:
    return ArtifactQualityIssue(
        issue_code=issue_code,
        severity=severity,
        pipeline_stage=pipeline_stage,
        document_id=document_id,
        page=page,
        block_id=(
            _string(block.get("block_id"))
            if block is not None
            else None
        ),
        block_ordinal=(
            _integer(block.get("ordinal"))
            if block is not None
            else None
        ),
        chunk_id=(
            _string(chunk.get("chunk_id"))
            if chunk is not None
            else None
        ),
        chunk_ordinal=(
            _integer(chunk.get("ordinal"))
            if chunk is not None
            else None
        ),
        evidence=evidence,
        remediation_hint=remediation_hint,
    )


def _has_invalid_page_range(
    row: dict[str, Any],
) -> bool:
    page_start = _integer(row.get("page_start"))
    page_end = _integer(row.get("page_end"))
    return (
        page_start is None
        or page_end is None
        or page_start < 1
        or page_end < page_start
    )


def _is_missing_image_target(
    *,
    image_path: str,
    artifacts_directory: Path,
) -> bool:
    candidate = Path(image_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return True

    root = artifacts_directory.resolve()
    target = (root / candidate).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        return True

    return not target.is_file()


def _is_very_short_text_block(
    block: dict[str, Any],
    text: str,
) -> bool:
    """Warn only on one-character body text, never semantic short terms."""

    return (
        block.get("block_type") == "paragraph"
        and len(text) == 1
        and text not in _ORPHAN_PUNCTUATION
    )


def _is_figure_label_candidate(
    block: dict[str, Any],
) -> bool:
    text = _string(block.get("text"))
    return (
        block.get("block_type") == "paragraph"
        and text is not None
        and bool(_FIGURE_LABEL.fullmatch(text.strip()))
    )


def _has_nearby_figure_caption(
    blocks: list[dict[str, Any]],
    start: int,
    end: int,
) -> bool:
    lower = max(0, start - 1)
    upper = min(len(blocks), end + 2)
    return any(
        blocks[index].get("block_type") == "figure_caption"
        for index in range(lower, upper)
    )


class ArtifactQualityAuditor:
    """Read artifact files and apply deterministic, conservative checks."""

    def __init__(
        self,
        *,
        inputs: ArtifactQualityInputs,
        config: AuditConfig | None = None,
    ) -> None:
        self._inputs = inputs
        self._config = config or AuditConfig()

    def audit(self) -> ArtifactQualityReport:
        inputs = self._inputs
        normalization_report = _read_json(
            inputs.normalization_report
        )
        chunking_report = _read_json(
            inputs.chunking_report
        )
        blocks = _read_jsonl(inputs.blocks)
        chunks = _read_jsonl(inputs.chunks)
        document_id = _document_id(
            normalization_report,
            blocks,
        )

        issues = [
            *self._audit_docling_markdown(
                document_id=document_id,
                normalization_report=normalization_report,
                blocks=blocks,
            ),
            *self._audit_normalization(
                document_id=document_id,
                report=normalization_report,
                blocks=blocks,
            ),
            *self._audit_chunking(
                document_id=document_id,
                report=chunking_report,
                blocks=blocks,
                chunks=chunks,
            ),
        ]

        return ArtifactQualityReport(
            document_id=document_id,
            inputs=inputs,
            config=self._config,
            issues=tuple(issues),
        )

    def _audit_docling_markdown(
        self,
        *,
        document_id: str | None,
        normalization_report: dict[str, Any],
        blocks: list[dict[str, Any]],
    ) -> list[ArtifactQualityIssue]:
        issues: list[ArtifactQualityIssue] = []
        overlay = normalization_report.get("correction_overlay")
        restorations = (
            overlay.get("formula_restorations", [])
            if isinstance(overlay, dict)
            else []
        )
        blocks_by_id = {
            _string(block.get("block_id")): block
            for block in blocks
            if _string(block.get("block_id")) is not None
        }
        lines = self._inputs.docling_markdown.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):
            if "formula-not-decoded" not in line:
                continue

            evidence = (
                "Docling Markdown contains "
                f"formula-not-decoded at line {line_number}"
            )
            matches = [
                item
                for item in restorations
                if isinstance(item, dict)
                and _integer(item.get("marker_line")) == line_number
            ]
            restored = False
            if len(matches) == 1:
                restoration = matches[0]
                equation = blocks_by_id.get(
                    _string(restoration.get("equation_block_id"))
                )
                restored = bool(
                    equation is not None
                    and equation.get("block_type") == "equation"
                    and _integer(equation.get("page_start"))
                    == _integer(restoration.get("marker_page"))
                    and _string(restoration.get("source_ref"))
                    and _string(restoration.get("correction_id"))
                )
            issues.append(
                _issue(
                    issue_code="FORMULA_NOT_DECODED",
                    severity=(
                        "warning" if restored else "error"
                    ),
                    pipeline_stage="conversion",
                    document_id=document_id,
                    evidence=evidence,
                    remediation_hint=(
                        "The source marker remains an upstream fact. "
                        "Verify that a reviewed correction overlay "
                        "restored the equation."
                    ),
                )
            )
            if restored:
                issues.append(
                    _issue(
                        issue_code=(
                            "FORMULA_RESTORED_BY_CORRECTION"
                        ),
                        severity="warning",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        evidence=(
                            f"{evidence}; normalization report records "
                            "a reviewed equation restoration"
                        ),
                        remediation_hint=(
                            "Keep the correction overlay source-anchored "
                            "and re-verify it against the original PDF "
                            "when the conversion changes."
                        ),
                    )
                )
            issues.append(
                _issue(
                    issue_code="TEXT_EXPECTS_MISSING_FORMULA",
                    severity="warning",
                    pipeline_stage="conversion",
                    document_id=document_id,
                    evidence=evidence,
                    remediation_hint=(
                        "Keep the upstream fact visible and record whether "
                        "a correction overlay restored the equation."
                    ),
                )
            )

        return issues

    def _audit_normalization(
        self,
        *,
        document_id: str | None,
        report: dict[str, Any],
        blocks: list[dict[str, Any]],
    ) -> list[ArtifactQualityIssue]:
        issues: list[ArtifactQualityIssue] = []
        expected_count = _integer(
            report.get("normalized_block_count")
        )

        if expected_count != len(blocks):
            issues.append(
                _issue(
                    issue_code="REPORT_COUNT_MISMATCH",
                    severity="error",
                    pipeline_stage="normalization",
                    document_id=document_id,
                    evidence=(
                        "normalization-report.json says "
                        f"normalized_block_count={expected_count}, "
                        f"but blocks.jsonl contains {len(blocks)} rows"
                    ),
                    remediation_hint=(
                        "Regenerate the complete normalization artifact "
                        "bundle together."
                    ),
                )
            )

        for block in blocks:
            text = _string(block.get("text"))
            page = _integer(block.get("page_start"))

            if text is None or not text.strip():
                issues.append(
                    _issue(
                        issue_code="EMPTY_BLOCK",
                        severity="error",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence="block text is empty or missing",
                        remediation_hint=(
                            "Correct the upstream conversion or remove the "
                            "empty source item during normalization."
                        ),
                    )
                )
                continue

            normalized_text = text.strip()
            if (
                block.get("block_type") in {
                    "paragraph",
                    "list_item",
                }
                and normalized_text in _ORPHAN_PUNCTUATION
            ):
                issues.append(
                    _issue(
                        issue_code="ORPHAN_PUNCTUATION_BLOCK",
                        severity="error",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "body block contains only "
                            f"{normalized_text!r}"
                        ),
                        remediation_hint=(
                            "Merge the punctuation into the intended text "
                            "through a reviewed correction overlay."
                        ),
                    )
                )

            if _has_invalid_page_range(block):
                issues.append(
                    _issue(
                        issue_code="INVALID_PAGE_RANGE",
                        severity="error",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "block page range is not a positive "
                            "nondecreasing interval"
                        ),
                        remediation_hint=(
                            "Repair page provenance in the conversion or "
                            "normalization stage."
                        ),
                    )
                )

            image_path = _string(block.get("image_path"))
            if image_path and _is_missing_image_target(
                image_path=image_path,
                artifacts_directory=self._inputs.blocks.parent,
            ):
                issues.append(
                    _issue(
                        issue_code="MISSING_IMAGE_TARGET",
                        severity="error",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "image_path does not resolve to a file inside "
                            f"the normalization artifact: {image_path}"
                        ),
                        remediation_hint=(
                            "Regenerate the bundle with its referenced image "
                            "assets, preserving relative image paths."
                        ),
                    )
                )

            if _is_very_short_text_block(block, normalized_text):
                issues.append(
                    _issue(
                        issue_code="VERY_SHORT_TEXT_BLOCK",
                        severity="warning",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "one-character paragraph is retained as body "
                            f"text: {normalized_text!r}"
                        ),
                        remediation_hint=(
                            "Review surrounding layout before changing it; "
                            "short semantic terms are not errors by default."
                        ),
                    )
                )

            if (
                block.get("block_type") == "paragraph"
                and _NUMBERED_HEADING.fullmatch(
                    normalized_text
                )
            ):
                issues.append(
                    _issue(
                        issue_code="HEADING_LIKE_PARAGRAPH",
                        severity="warning",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "numbered paragraph resembles a heading: "
                            f"{normalized_text!r}"
                        ),
                        remediation_hint=(
                            "Compare with the source page before changing "
                            "heading_path or block type."
                        ),
                    )
                )

            if _SUSPICIOUS_SPACING.search(normalized_text):
                issues.append(
                    _issue(
                        issue_code="SUSPICIOUS_WORD_SPACING",
                        severity="warning",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        page=page,
                        block=block,
                        evidence=(
                            "URL-like or dotted Latin text contains "
                            "an internal space"
                        ),
                        remediation_hint=(
                            "Apply only a source-confirmed, semantic-safe "
                            "normalization correction."
                        ),
                    )
                )

        issues.extend(
            self._audit_block_sequences(
                document_id=document_id,
                blocks=blocks,
            )
        )
        issues.extend(
            self._audit_normalization_statistics(
                document_id=document_id,
                report=report,
            )
        )
        return issues

    def _audit_block_sequences(
        self,
        *,
        document_id: str | None,
        blocks: list[dict[str, Any]],
    ) -> list[ArtifactQualityIssue]:
        issues: list[ArtifactQualityIssue] = []
        by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for block in blocks:
            page = _integer(block.get("page_start"))
            if page is not None:
                by_page[page].append(block)

        for page, page_blocks in sorted(by_page.items()):
            start = 0
            while start < len(page_blocks):
                if not _is_figure_label_candidate(
                    page_blocks[start]
                ):
                    start += 1
                    continue

                end = start + 1
                while (
                    end < len(page_blocks)
                    and _is_figure_label_candidate(
                        page_blocks[end]
                    )
                ):
                    end += 1

                if (
                    end - start >= 2
                    and _has_nearby_figure_caption(
                        page_blocks,
                        start,
                        end,
                    )
                ):
                    labels = " ".join(
                        _string(block.get("text")) or ""
                        for block in page_blocks[start:end]
                    )
                    issues.append(
                        _issue(
                            issue_code=(
                                "POSSIBLE_FIGURE_LABEL_SEQUENCE"
                            ),
                            severity="warning",
                            pipeline_stage="normalization",
                            document_id=document_id,
                            page=page,
                            block=page_blocks[start],
                            evidence=(
                                "adjacent short labels near a figure caption: "
                                f"{labels!r}"
                            ),
                            remediation_hint=(
                                "Classify confirmed graphical labels as "
                                "non-indexable rather than deleting all short "
                                "text."
                            ),
                        )
                    )

                start = end

        for previous, current in zip(blocks, blocks[1:]):
            previous_text = _string(previous.get("text"))
            current_text = _string(current.get("text"))
            if (
                previous_text is None
                or current_text is None
                or previous.get("page_start")
                != current.get("page_start")
                or previous_text.rstrip()[-1:]
                in _SENTENCE_TERMINATORS
                or current_text.strip()
                not in _ORPHAN_PUNCTUATION
            ):
                continue

            issues.append(
                _issue(
                    issue_code="POSSIBLE_READING_ORDER_BREAK",
                    severity="warning",
                    pipeline_stage="normalization",
                    document_id=document_id,
                    page=_integer(previous.get("page_start")),
                    block=previous,
                    evidence=(
                        "unfinished body text is immediately followed by "
                        f"an orphan punctuation block {current_text!r}"
                    ),
                    remediation_hint=(
                        "Review the source page and apply a narrowly scoped "
                        "reordering or merge correction if confirmed."
                    ),
                )
            )

        return issues

    def _audit_normalization_statistics(
        self,
        *,
        document_id: str | None,
        report: dict[str, Any],
    ) -> list[ArtifactQualityIssue]:
        issues: list[ArtifactQualityIssue] = []
        short_fragment_ratio = report.get("short_fragment_ratio")

        if (
            isinstance(short_fragment_ratio, (int, float))
            and not isinstance(short_fragment_ratio, bool)
            and short_fragment_ratio
            > self._config.short_fragment_ratio_warning
        ):
            issues.append(
                _issue(
                    issue_code="HIGH_SHORT_FRAGMENT_RATIO",
                    severity="warning",
                    pipeline_stage="normalization",
                    document_id=document_id,
                    evidence=(
                        "short_fragment_ratio="
                        f"{short_fragment_ratio:.6f} exceeds "
                        f"{self._config.short_fragment_ratio_warning:.6f}"
                    ),
                    remediation_hint=(
                        "Sample the flagged blocks and fix only confirmed "
                        "conversion or reading-order defects."
                    ),
                )
            )

        source_pages = report.get("source_pages")
        review_pages = report.get("pages_requiring_review")
        if (
            isinstance(source_pages, list)
            and source_pages
            and isinstance(review_pages, list)
        ):
            ratio = len(review_pages) / len(source_pages)
            if ratio > self._config.review_page_ratio_warning:
                issues.append(
                    _issue(
                        issue_code="EXCESSIVE_REVIEW_PAGE_RATIO",
                        severity="warning",
                        pipeline_stage="normalization",
                        document_id=document_id,
                        evidence=(
                            f"{len(review_pages)}/{len(source_pages)} "
                            "source pages require review "
                            f"({ratio:.6f})"
                        ),
                        remediation_hint=(
                            "Prioritize source-confirmed repairs; do not "
                            "treat review markers as automatic errors."
                        ),
                    )
                )

        return issues

    def _audit_chunking(
        self,
        *,
        document_id: str | None,
        report: dict[str, Any],
        blocks: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> list[ArtifactQualityIssue]:
        issues: list[ArtifactQualityIssue] = []
        expected_input_count = _integer(
            report.get("input_block_count")
        )
        expected_output_count = _integer(
            report.get("output_chunk_count")
        )

        if expected_input_count != len(blocks):
            issues.append(
                _issue(
                    issue_code="REPORT_COUNT_MISMATCH",
                    severity="error",
                    pipeline_stage="chunking",
                    document_id=document_id,
                    evidence=(
                        "chunking-report.json says "
                        f"input_block_count={expected_input_count}, "
                        f"but blocks.jsonl contains {len(blocks)} rows"
                    ),
                    remediation_hint=(
                        "Regenerate chunking from the exact normalized "
                        "artifact bundle."
                    ),
                )
            )

        if expected_output_count != len(chunks):
            issues.append(
                _issue(
                    issue_code="REPORT_COUNT_MISMATCH",
                    severity="error",
                    pipeline_stage="chunking",
                    document_id=document_id,
                    evidence=(
                        "chunking-report.json says "
                        f"output_chunk_count={expected_output_count}, "
                        f"but chunks.jsonl contains {len(chunks)} rows"
                    ),
                    remediation_hint=(
                        "Regenerate the complete chunking artifact bundle "
                        "together."
                    ),
                )
            )

        block_ids = {
            block_id
            for block in blocks
            if (
                block_id := _string(
                    block.get("block_id")
                )
            )
        }

        for chunk in chunks:
            page = _integer(chunk.get("page_start"))
            content = _string(chunk.get("content"))
            index_text = _string(chunk.get("index_text"))

            if (
                content is None
                or not content.strip()
                or index_text is None
                or not index_text.strip()
            ):
                issues.append(
                    _issue(
                        issue_code="EMPTY_CHUNK",
                        severity="error",
                        pipeline_stage="chunking",
                        document_id=document_id,
                        page=page,
                        chunk=chunk,
                        evidence=(
                            "chunk content or index_text is empty or missing"
                        ),
                        remediation_hint=(
                            "Fix the normalized input and regenerate chunks; "
                            "do not hand-edit chunks.jsonl."
                        ),
                    )
                )

            if _has_invalid_page_range(chunk):
                issues.append(
                    _issue(
                        issue_code="INVALID_PAGE_RANGE",
                        severity="error",
                        pipeline_stage="chunking",
                        document_id=document_id,
                        page=page,
                        chunk=chunk,
                        evidence=(
                            "chunk page range is not a positive "
                            "nondecreasing interval"
                        ),
                        remediation_hint=(
                            "Repair source provenance in normalized blocks "
                            "and regenerate chunks."
                        ),
                    )
                )

            if (
                index_text is not None
                and len(index_text)
                > self._config.max_chunk_chars
            ):
                issues.append(
                    _issue(
                        issue_code="CHUNK_EXCEEDS_CONFIGURED_LIMIT",
                        severity="error",
                        pipeline_stage="chunking",
                        document_id=document_id,
                        page=page,
                        chunk=chunk,
                        evidence=(
                            f"index_text has {len(index_text)} characters; "
                            f"configured limit is "
                            f"{self._config.max_chunk_chars}"
                        ),
                        remediation_hint=(
                            "Adjust Chunker configuration or split eligible "
                            "normalized content before regenerating."
                        ),
                    )
                )

            references = chunk.get("block_ids")
            if not isinstance(references, list):
                references = []

            broken_references = [
                reference
                for reference in references
                if not isinstance(reference, str)
                or reference not in block_ids
            ]
            seen_references: list[object] = []
            duplicate_references = any(
                reference in seen_references
                or seen_references.append(reference)
                for reference in references
            )
            if broken_references or duplicate_references:
                details: list[str] = []
                if broken_references:
                    details.append(
                        "unknown block IDs: "
                        + ", ".join(
                            repr(reference)
                            for reference in broken_references
                        )
                    )
                if duplicate_references:
                    details.append("duplicate block IDs")

                issues.append(
                    _issue(
                        issue_code="BROKEN_BLOCK_REFERENCE",
                        severity="error",
                        pipeline_stage="chunking",
                        document_id=document_id,
                        page=page,
                        chunk=chunk,
                        evidence="; ".join(details),
                        remediation_hint=(
                            "Regenerate chunks from the matching blocks.jsonl "
                            "and preserve ordered, unique source references."
                        ),
                    )
                )

        return issues


def audit_artifacts(
    *,
    inputs: ArtifactQualityInputs,
    config: AuditConfig | None = None,
) -> ArtifactQualityReport:
    """Audit one artifact bundle through the public quality API."""

    return ArtifactQualityAuditor(
        inputs=inputs,
        config=config,
    ).audit()
