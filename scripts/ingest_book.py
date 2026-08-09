"""Build a versioned, auditable multi-section corpus from one PDF."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rag_lab.ingestion import (
    process_book,
    read_book_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a manifest-defined book into independently audited "
            "sections and an assembled retrieval corpus."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--section",
        action="append",
        dest="section_ids",
        help=(
            "Build one named section. May be repeated. Omitting this "
            "builds the complete manifest and assembles corpus/chunks.jsonl."
        ),
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        manifest = read_book_manifest(arguments.manifest)
        results = process_book(
            manifest=manifest,
            source_pdf=arguments.source,
            output_root=arguments.output,
            section_ids=arguments.section_ids,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
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
    if not sys.flags.utf8_mode:
        raise SystemExit(
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
                check=False,
            ).returncode
        )
    raise SystemExit(main())
