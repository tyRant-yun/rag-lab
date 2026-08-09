from __future__ import annotations

import json
import shutil
from pathlib import Path

from rag_lab.contracts.blocks import (
    BlockType,
    NormalizedBlock,
)
from rag_lab.normalization.models import (
    NormalizationResult,
)


def write_normalization_outputs(
    *,
    result: NormalizationResult,
    output_directory: Path,
    asset_source_directory: Path | None = None,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if asset_source_directory is None:
        _validate_referenced_images(
            blocks=result.blocks,
            output_directory=output_directory,
        )
    else:
        _copy_referenced_images(
            blocks=result.blocks,
            source_directory=(
                asset_source_directory
            ),
            output_directory=output_directory,
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


def _validate_referenced_images(
    *,
    blocks: tuple[NormalizedBlock, ...],
    output_directory: Path,
) -> None:
    output_root = output_directory.resolve()
    relative_paths = {
        block.image_path
        for block in blocks
        if block.image_path
    }

    for image_path in sorted(relative_paths):
        target = (
            output_root / image_path
        ).resolve()

        try:
            target.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                "image_path escapes the output "
                "directory"
            ) from error

        if not target.is_file():
            raise FileNotFoundError(
                "asset_source_directory is required "
                "when a referenced image is not "
                f"already in the output directory: "
                f"{target}"
            )


def _copy_referenced_images(
    *,
    blocks: tuple[NormalizedBlock, ...],
    source_directory: Path,
    output_directory: Path,
) -> None:
    source_root = source_directory.resolve()
    output_root = output_directory.resolve()
    relative_paths = {
        block.image_path
        for block in blocks
        if block.image_path
    }

    for image_path in sorted(relative_paths):
        relative_path = Path(image_path)

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                "image_path must be relative to "
                "the document artifact directory"
            )

        source = (
            source_root / relative_path
        ).resolve()
        target = (
            output_root / relative_path
        ).resolve()

        try:
            source.relative_to(source_root)
            target.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                "image_path escapes its artifact "
                "directory"
            ) from error

        if not source.is_file():
            raise FileNotFoundError(
                f"referenced image not found: "
                f"{source}"
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if source != target:
            shutil.copy2(source, target)


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
            f"type={block.block_type} "
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
        BlockType.DOCUMENT_TITLE.value,
        BlockType.SECTION_HEADING.value,
    }:
        level = min(
            max(len(block.heading_path), 1),
            6,
        )
        return f"{'#' * level} {block.text}"

    if block.block_type == (
        BlockType.FIGURE_CAPTION.value
    ):
        if block.image_path:
            return (
                f"![{block.text}]"
                f"({block.image_path})"
            )

        return f"*{block.text}*"

    if block.block_type == BlockType.FIGURE_LABEL.value:
        return (
            "<!-- non-indexable figure label: "
            f"{block.text} -->"
        )

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
