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

_SENTENCE_TERMINATORS: frozenset[str] = frozenset(
    {
        "。",
        "！",
        "？",
        ".",
        "!",
        "?",
    }
)

_TRAILING_CLOSERS: frozenset[str] = frozenset(
    {
        "”",
        "’",
        '"',
        "'",
        "）",
        ")",
        "]",
        "】",
        "》",
        "〉",
    }
)


@dataclass(frozen=True, slots=True)
class _CandidateGroup:
    """Consecutive body blocks under one heading path."""

    heading_path: tuple[str, ...]
    blocks: tuple[NormalizedBlock, ...]


@dataclass(frozen=True, slots=True)
class _ContentUnit:
    """Text assembled from one or more source blocks."""

    text: str
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


def _ends_complete_sentence(
    text: str,
) -> bool:
    """Check terminal punctuation, ignoring closing quotes."""

    candidate = text.rstrip()

    while (
        candidate
        and candidate[-1] in _TRAILING_CLOSERS
    ):
        candidate = candidate[:-1].rstrip()

    return bool(
        candidate
        and candidate[-1] in _SENTENCE_TERMINATORS
    )


def _is_cross_page_paragraph_continuation(
    previous: NormalizedBlock,
    current: NormalizedBlock,
) -> bool:
    """Return whether two blocks likely form one paragraph."""

    return (
        previous.block_type
        == BlockType.PARAGRAPH.value
        and current.block_type
        == BlockType.PARAGRAPH.value
        and previous.document_id
        == current.document_id
        and previous.heading_path
        == current.heading_path
        and current.ordinal
        == previous.ordinal + 1
        and current.page_start
        == previous.page_end + 1
        and not _ends_complete_sentence(
            previous.text
        )
    )


def _join_continuation_text(
    previous: str,
    current: str,
) -> str:
    """Join continuation text without losing characters."""

    left = previous.rstrip()
    right = current.lstrip()

    left_boundary = left[-1]
    right_boundary = right[0]

    if (
        left_boundary.isascii()
        and left_boundary.isalnum()
        and right_boundary.isascii()
        and right_boundary.isalnum()
    ):
        return f"{left} {right}"

    return f"{left}{right}"


def _merge_cross_page_paragraphs(
    group: _CandidateGroup,
) -> tuple[tuple[_ContentUnit, ...], int]:
    """Merge likely cross-page paragraph continuations."""

    units: list[_ContentUnit] = []
    join_count = 0

    for block in group.blocks:
        if (
            units
            and _is_cross_page_paragraph_continuation(
                units[-1].blocks[-1],
                block,
            )
        ):
            previous_unit = units.pop()

            units.append(
                _ContentUnit(
                    text=_join_continuation_text(
                        previous_unit.text,
                        block.text,
                    ),
                    blocks=(
                        previous_unit.blocks
                        + (block,)
                    ),
                )
            )

            join_count += 1
            continue

        units.append(
            _ContentUnit(
                text=block.text.strip(),
                blocks=(block,),
            )
        )

    return tuple(units), join_count
