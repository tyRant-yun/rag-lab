from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.contracts import (
    BlockType,
    NormalizedBlock,
)

_CONTROL_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        BlockType.DOCUMENT_TITLE.value,
        BlockType.SECTION_HEADING.value,
    }
)

_BODY_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        BlockType.PARAGRAPH.value,
        BlockType.LIST_ITEM.value,
        BlockType.FIGURE_CAPTION.value,
        BlockType.TABLE.value,
        BlockType.CODE.value,
        BlockType.EQUATION.value,
    }
)


@dataclass(frozen=True, slots=True)
class _CandidateGroup:
    """Consecutive body blocks under one heading path."""

    heading_path: tuple[str, ...]
    blocks: tuple[NormalizedBlock, ...]


def _is_control_block(
    block: NormalizedBlock,
) -> bool:
    """Return whether a block controls chunk boundaries."""

    return block.block_type in _CONTROL_BLOCK_TYPES


def _is_body_block(
    block: NormalizedBlock,
) -> bool:
    """Return whether a block contributes chunk content."""

    return block.block_type in _BODY_BLOCK_TYPES


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


def _group_body_blocks(
    blocks: Sequence[NormalizedBlock],
) -> list[_CandidateGroup]:
    """Group consecutive body blocks by heading path."""

    ordered_blocks = _validate_and_sort_blocks(
        blocks
    )

    groups: list[_CandidateGroup] = []
    current_heading_path: tuple[str, ...] = ()
    current_blocks: list[NormalizedBlock] = []

    def flush_current_group() -> None:
        nonlocal current_heading_path
        nonlocal current_blocks

        if not current_blocks:
            return

        groups.append(
            _CandidateGroup(
                heading_path=current_heading_path,
                blocks=tuple(current_blocks),
            )
        )

        current_heading_path = ()
        current_blocks = []

    for block in ordered_blocks:
        if _is_control_block(block):
            flush_current_group()
            continue

        if not _is_body_block(block):
            raise ValueError(
                f"unsupported block role: "
                f"{block.block_type}"
            )

        block_heading_path = tuple(
            block.heading_path
        )

        if (
            current_blocks
            and block_heading_path
            != current_heading_path
        ):
            flush_current_group()

        if not current_blocks:
            current_heading_path = (
                block_heading_path
            )

        current_blocks.append(block)

    flush_current_group()

    return groups
