from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from rag_lab.contracts import KnowledgeChunk


def read_knowledge_chunks_jsonl(
    path: Path,
) -> tuple[KnowledgeChunk, ...]:
    """Read and strictly validate KnowledgeChunk JSONL records."""

    chunks: list[KnowledgeChunk] = []

    with path.open(
        encoding="utf-8",
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
                chunk = KnowledgeChunk.model_validate(
                    payload,
                    strict=True,
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    "invalid KnowledgeChunk"
                ) from error

            chunks.append(chunk)

    return tuple(chunks)
