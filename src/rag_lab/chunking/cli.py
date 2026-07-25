from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag_lab.chunking import (
    ChunkingConfig,
    chunk_normalized_blocks,
)
from rag_lab.chunking.serialization import (
    read_normalized_blocks_jsonl,
    write_chunking_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert normalized blocks into "
            "source-grounded knowledge chunks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to blocks.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for chunking outputs.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--chunking-version",
        default="1.0.0",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        blocks = read_normalized_blocks_jsonl(
            arguments.input
        )

        config = ChunkingConfig(
            max_chars=arguments.max_chars,
            chunking_version=(
                arguments.chunking_version
            ),
        )

        result = chunk_normalized_blocks(
            blocks=blocks,
            config=config,
        )

        write_chunking_outputs(
            result=result,
            output_directory=arguments.output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    report = result.report

    print(
        f"chunked {report.input_block_count} "
        f"blocks into "
        f"{report.output_chunk_count} chunks"
    )
    print(
        f"cross-page joins: "
        f"{report.cross_page_join_count}"
    )
    print(
        f"long blocks split: "
        f"{report.long_block_split_count}"
    )
    print(
        f"oversized atomic blocks: "
        f"{report.oversized_atomic_block_count}"
    )
    print(
        f"output: {arguments.output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
