from __future__ import annotations

from collections.abc import Sequence

from rag_lab.contracts import NormalizedBlock


def _validate_and_sort_blocks(
    blocks: Sequence[NormalizedBlock],
) -> list[NormalizedBlock]:
    """Validate one document and return blocks in ordinal order."""

    if not blocks:
        raise ValueError("blocks cannot be empty")

    if any(
        not isinstance(block, NormalizedBlock)
        for block in blocks
    ):
        raise TypeError(
            "all blocks must be NormalizedBlock instances"
        )

    document_ids = {
        block.document_id
        for block in blocks
    }
    if len(document_ids) != 1:
        raise ValueError(
            "blocks must belong to one document"
        )

    source_paths = {
        block.source_path
        for block in blocks
    }
    if len(source_paths) != 1:
        raise ValueError(
            "blocks must have one source_path"
        )

    normalization_versions = {
        block.normalization_version
        for block in blocks
    }
    if len(normalization_versions) != 1:
        raise ValueError(
            "blocks must have one normalization_version"
        )

    block_ids = [
        block.block_id
        for block in blocks
    ]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(
            "block_id values must be unique"
        )

    ordinals = [
        block.ordinal
        for block in blocks
    ]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(
            "ordinal values must be unique"
        )

    return sorted(
        blocks,
        key=lambda block: block.ordinal,
    )
