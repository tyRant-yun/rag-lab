from __future__ import annotations

import json
from pathlib import Path

from knowledge_normalizer.models import (
    BlockType,
    NormalizationResult,
    NormalizedBlock,
)


def write_normalization_outputs(
    *,
    result: NormalizationResult,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_jsonl(
        result.blocks,
        output_directory / "blocks.jsonl",
    )
    _write_markdown(
        result.blocks,
        output_directory / "document.md",
    )
    _write_report(
        result,
        output_directory
        / "normalization-report.json",
    )


def _write_jsonl(
    blocks: tuple[NormalizedBlock, ...],
    path: Path,
) -> None:
    lines = [
        json.dumps(
            block.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for block in blocks
    ]
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_markdown(
    blocks: tuple[NormalizedBlock, ...],
    path: Path,
) -> None:
    sections: list[str] = []

    for block in blocks:
        metadata = (
            f"<!-- ordinal={block.ordinal} "
            f"type={block.block_type.value} "
            f"pages={block.page_start}-"
            f"{block.page_end} -->"
        )
        body = _render_markdown_block(block)
        sections.append(
            f"{metadata}\n\n{body}"
        )

    path.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _render_markdown_block(
    block: NormalizedBlock,
) -> str:
    if block.block_type in {
        BlockType.DOCUMENT_TITLE,
        BlockType.SECTION_HEADING,
    }:
        level = min(
            max(len(block.heading_path), 1),
            6,
        )
        return f"{'#' * level} {block.text}"

    if block.block_type == (
        BlockType.FIGURE_CAPTION
    ):
        return f"*{block.text}*"

    return block.text


def _write_report(
    result: NormalizationResult,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            result.report.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

