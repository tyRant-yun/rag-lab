from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from rag_lab.chunking.models import (
    ChunkingResult,
)
from rag_lab.contracts import (
    KnowledgeChunk,
    NormalizedBlock,
)


def read_normalized_blocks_jsonl(
    path: Path,
) -> tuple[NormalizedBlock, ...]:
    """Read and validate normalized blocks line by line."""

    blocks: list[NormalizedBlock] = []

    with path.open(
        encoding="utf-8"
    ) as stream:
        for line_number, raw_line in enumerate(
            stream,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "empty JSONL record"
                )

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "invalid JSON"
                ) from error

            try:
                block = NormalizedBlock.model_validate(
                    payload,
                    strict=True,
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "invalid NormalizedBlock"
                ) from error

            blocks.append(block)

    return tuple(blocks)

def write_chunking_outputs(
    *,
    result: ChunkingResult,
    output_directory: Path,
) -> None:
    """Write machine and human review artifacts."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_chunks_jsonl(
        result.chunks,
        output_directory / "chunks.jsonl",
    )
    _write_chunks_markdown(
        result.chunks,
        output_directory / "chunks.md",
    )
    _write_chunking_report(
        result,
        output_directory
        / "chunking-report.json",
    )


def _write_chunks_jsonl(
    chunks: Sequence[KnowledgeChunk],
    path: Path,
) -> None:
    lines = [
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]

    content = "\n".join(lines)

    if lines:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def _write_chunks_markdown(
    chunks: Sequence[KnowledgeChunk],
    path: Path,
) -> None:
    sections = [
        _render_chunk_review(chunk)
        for chunk in chunks
    ]

    content = "\n\n---\n\n".join(
        sections
    )

    if sections:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def _render_chunk_review(
    chunk: KnowledgeChunk,
) -> str:
    heading_path = " > ".join(
        chunk.heading_path
    )
    block_ids = ", ".join(
        f"`{block_id}`"
        for block_id in chunk.block_ids
    )

    metadata = (
        f"<!-- ordinal={chunk.ordinal} "
        f"pages={chunk.page_start}-"
        f"{chunk.page_end} "
        f"content_chars={len(chunk.content)} "
        f"index_chars={len(chunk.index_text)} -->"
    )

    return (
        f"{metadata}\n\n"
        f"## Chunk {chunk.ordinal}\n\n"
        f"**Chunk ID:** `{chunk.chunk_id}`\n\n"
        f"**Heading path:** {heading_path}\n\n"
        f"**Source pages:** "
        f"{chunk.page_start}-{chunk.page_end}\n\n"
        f"**Block IDs:** {block_ids}\n\n"
        f"{chunk.content}"
    )


def _write_chunking_report(
    result: ChunkingResult,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            result.report.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
