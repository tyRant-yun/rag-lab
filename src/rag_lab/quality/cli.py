from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_lab.quality.auditor import audit_artifacts
from rag_lab.quality.models import (
    ArtifactQualityInputs,
    AuditConfig,
)
from rag_lab.quality.serialization import (
    write_artifact_quality_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit-artifacts",
        description=(
            "Run deterministic, read-only quality checks over a "
            "normalization and chunking artifact bundle."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Artifact root containing normalized/, one chunked-*/ "
            "directory, and the Docling Markdown source."
        ),
    )
    parser.add_argument(
        "--docling-markdown",
        type=Path,
        help="Docling-generated Markdown to audit.",
    )
    parser.add_argument(
        "--normalization-report",
        type=Path,
        help="Path to normalization-report.json.",
    )
    parser.add_argument(
        "--blocks",
        type=Path,
        help="Path to normalized blocks.jsonl.",
    )
    parser.add_argument(
        "--chunking-report",
        type=Path,
        help="Path to chunking-report.json.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        help="Path to chunks.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON report path. Defaults to "
            "artifact-quality-report.json under --artifact-root, or the "
            "current directory."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Configured maximum index_text length. Default: 1200.",
    )
    parser.add_argument(
        "--short-fragment-ratio-warning",
        type=float,
        default=0.04,
        help="Warning threshold for short_fragment_ratio. Default: 0.04.",
    )
    parser.add_argument(
        "--review-page-ratio-warning",
        type=float,
        default=0.25,
        help=(
            "Warning threshold for review-page ratio. Default: 0.25."
        ),
    )
    return parser


def _resolve_docling_markdown(root: Path) -> Path:
    candidates = sorted(root.glob("*.md"))
    if len(candidates) != 1:
        raise ValueError(
            "--artifact-root must contain exactly one direct Docling "
            "Markdown file, or pass --docling-markdown explicitly"
        )

    return candidates[0]


def _resolve_chunk_directory(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("chunked-*")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            "--artifact-root must contain exactly one chunked-* "
            "directory, or pass --chunking-report and --chunks explicitly"
        )

    return candidates[0]


def resolve_inputs(
    arguments: argparse.Namespace,
) -> ArtifactQualityInputs:
    """Resolve explicit CLI paths, with deterministic artifact-root defaults."""

    root = arguments.artifact_root
    if root is not None:
        root = root.resolve()
        normalized = root / "normalized"
        chunked = _resolve_chunk_directory(root)
        docling_markdown = (
            arguments.docling_markdown
            or _resolve_docling_markdown(root)
        )
        normalization_report = (
            arguments.normalization_report
            or normalized / "normalization-report.json"
        )
        blocks = arguments.blocks or normalized / "blocks.jsonl"
        chunking_report = (
            arguments.chunking_report
            or chunked / "chunking-report.json"
        )
        chunks = arguments.chunks or chunked / "chunks.jsonl"
    else:
        required = {
            "--docling-markdown": arguments.docling_markdown,
            "--normalization-report": arguments.normalization_report,
            "--blocks": arguments.blocks,
            "--chunking-report": arguments.chunking_report,
            "--chunks": arguments.chunks,
        }
        missing = [
            option
            for option, value in required.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "missing required artifact paths: "
                + ", ".join(missing)
            )

        docling_markdown = arguments.docling_markdown
        normalization_report = arguments.normalization_report
        blocks = arguments.blocks
        chunking_report = arguments.chunking_report
        chunks = arguments.chunks

    return ArtifactQualityInputs(
        docling_markdown=Path(docling_markdown),
        normalization_report=Path(normalization_report),
        blocks=Path(blocks),
        chunking_report=Path(chunking_report),
        chunks=Path(chunks),
    )


def _output_path(
    arguments: argparse.Namespace,
) -> Path:
    if arguments.output is not None:
        return arguments.output

    if arguments.artifact_root is not None:
        return (
            arguments.artifact_root
            / "artifact-quality-report.json"
        )

    return Path("artifact-quality-report.json")


def _render_summary(report_path: Path, report) -> str:
    lines = [
        "Artifact quality audit",
        f"Errors: {report.error_count}",
        f"Warnings: {report.warning_count}",
        f"Status: {'passed' if report.passed else 'failed'}",
        f"Report: {report_path.resolve()}",
    ]

    for issue in report.issues:
        location = []
        if issue.page is not None:
            location.append(f"page={issue.page}")
        if issue.block_ordinal is not None:
            location.append(f"block={issue.block_ordinal}")
        if issue.chunk_ordinal is not None:
            location.append(f"chunk={issue.chunk_ordinal}")
        suffix = "" if not location else " (" + ", ".join(location) + ")"
        lines.append(
            f"[{issue.severity}] {issue.issue_code}{suffix}"
        )

    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)

    try:
        inputs = resolve_inputs(arguments)
        config = AuditConfig(
            max_chunk_chars=arguments.max_chars,
            short_fragment_ratio_warning=(
                arguments.short_fragment_ratio_warning
            ),
            review_page_ratio_warning=(
                arguments.review_page_ratio_warning
            ),
        )
        report = audit_artifacts(
            inputs=inputs,
            config=config,
        )
        report_path = _output_path(arguments)
        write_artifact_quality_report(
            report=report,
            output_path=report_path,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(_render_summary(report_path, report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
