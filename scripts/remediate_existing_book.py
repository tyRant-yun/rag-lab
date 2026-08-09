"""Rebuild a full corpus from an immutable prior Docling conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.ingestion import (
    process_existing_docling_book,
    read_book_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse existing Docling JSON/Markdown, apply source-bound "
            "correction overlays, then rerun normalization, chunking and "
            "quality gates."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--existing-docling-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        manifest = read_book_manifest(arguments.manifest)
        results = process_existing_docling_book(
            manifest=manifest,
            source_pdf=arguments.source,
            existing_docling_root=arguments.existing_docling_root,
            correction_root=arguments.correction_root,
            output_root=arguments.output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    for result in results:
        print(
            f"{result.section.section_id}: "
            f"chunks={result.chunk_count} "
            f"errors={result.error_count} "
            f"warnings={result.warning_count}"
        )
    print(f"output: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
