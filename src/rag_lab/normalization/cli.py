from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rag_lab.normalization.normalizer import (
    normalize_docling_document,
)
from rag_lab.normalization.serialization import (
    write_normalization_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Docling JSON into ordered "
            "source-grounded blocks."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--normalization-version",
        default="1.0.0",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = build_parser().parse_args(
        argv
    )

    with arguments.input_json.open(
        encoding="utf-8"
    ) as stream:
        document = json.load(stream)

    result = normalize_docling_document(
        docling_document=document,
        source_path=arguments.source,
        normalization_version=(
            arguments.normalization_version
        ),
    )
    write_normalization_outputs(
        result=result,
        output_directory=arguments.output,
    )

    print(
        f"normalized {len(result.blocks)} blocks "
        f"from {len(result.report.source_pages)} "
        "pages"
    )
    print(
        f"reordered blocks: "
        f"{result.report.reordered_block_count}"
    )
    print(
        f"output: {arguments.output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
