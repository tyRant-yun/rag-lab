from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.chunking.models import ChunkingConfig
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

_ATOMIC_BODY_TYPES: frozenset[str] = frozenset(
    {
        BlockType.TABLE.value,
        BlockType.CODE.value,
        BlockType.EQUATION.value,
    }
)

_SPLITTABLE_BODY_TYPES: frozenset[str] = frozenset(
    {
        BlockType.PARAGRAPH.value,
        BlockType.LIST_ITEM.value,
        BlockType.FIGURE_CAPTION.value,
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


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    """Content units waiting to become a KnowledgeChunk."""

    heading_path: tuple[str, ...]
    units: tuple[_ContentUnit, ...]


@dataclass(frozen=True, slots=True)
class _UnitPreparationResult:
    """Prepared units and oversized-content statistics."""

    units: tuple[_ContentUnit, ...]
    long_block_split_count: int
    oversized_atomic_block_count: int


@dataclass(frozen=True, slots=True)
class _DraftPreparationResult:
    """Chunk drafts and aggregated processing statistics."""

    drafts: tuple[_ChunkDraft, ...]
    cross_page_join_count: int
    long_block_split_count: int
    oversized_atomic_block_count: int


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


def _render_draft_content(
    draft: _ChunkDraft,
) -> str:
    """Render body text while preserving unit boundaries."""

    return "\n\n".join(
        unit.text
        for unit in draft.units
    )


def _render_draft_index_text(
    draft: _ChunkDraft,
) -> str:
    """Render the exact text used for size and retrieval."""

    heading_text = "\n".join(
        draft.heading_path
    )
    content = _render_draft_content(draft)

    if not content:
        return heading_text

    return f"{heading_text}\n\n{content}"


def _pack_content_units(
    *,
    heading_path: tuple[str, ...],
    units: Sequence[_ContentUnit],
    config: ChunkingConfig,
) -> tuple[_ChunkDraft, ...]:
    """Greedily pack ordered units under max_chars."""

    drafts: list[_ChunkDraft] = []
    current_units: list[_ContentUnit] = []

    for unit in units:
        candidate = _ChunkDraft(
            heading_path=heading_path,
            units=tuple(
                current_units + [unit]
            ),
        )

        if (
            current_units
            and len(
                _render_draft_index_text(
                    candidate
                )
            )
            > config.max_chars
        ):
            drafts.append(
                _ChunkDraft(
                    heading_path=heading_path,
                    units=tuple(current_units),
                )
            )
            current_units = [unit]
            continue

        current_units.append(unit)

    if current_units:
        drafts.append(
            _ChunkDraft(
                heading_path=heading_path,
                units=tuple(current_units),
            )
        )

    return tuple(drafts)


def _sentence_boundary_positions(
    text: str,
) -> tuple[int, ...]:
    """Return end offsets for conservative sentence boundaries."""

    boundaries: list[int] = []
    index = 0

    while index < len(text):
        character = text[index]

        if character not in _SENTENCE_TERMINATORS:
            index += 1
            continue

        end = index + 1

        while (
            end < len(text)
            and text[end] in _TRAILING_CLOSERS
        ):
            end += 1

        # 英文句点只有位于文本末尾或后接空白时，
        # 才视为句子边界，避免拆开 IP、域名和小数。
        if (
            character == "."
            and end < len(text)
            and not text[end].isspace()
        ):
            index += 1
            continue

        boundaries.append(end)
        index = end

    return tuple(boundaries)


def _split_text_to_budget(
    text: str,
    max_chars: int,
) -> tuple[str, ...]:
    """Split text at sentence boundaries with hard fallback."""

    if max_chars < 1:
        raise ValueError(
            "max_chars must leave room for content"
        )

    normalized = text.strip()

    if len(normalized) <= max_chars:
        return (normalized,)

    boundaries = _sentence_boundary_positions(
        normalized
    )

    fragments: list[str] = []
    start = 0

    while start < len(normalized):
        limit = min(
            start + max_chars,
            len(normalized),
        )

        if limit == len(normalized):
            end = limit
        else:
            eligible_boundaries = [
                boundary
                for boundary in boundaries
                if start < boundary <= limit
            ]

            end = (
                eligible_boundaries[-1]
                if eligible_boundaries
                else limit
            )

        fragment = normalized[start:end].strip()

        if fragment:
            fragments.append(fragment)

        start = end

        while (
            start < len(normalized)
            and normalized[start].isspace()
        ):
            start += 1

    return tuple(fragments)


def _content_char_budget(
    *,
    heading_path: tuple[str, ...],
    config: ChunkingConfig,
) -> int:
    """Return content capacity after heading context."""

    heading_text = "\n".join(heading_path)

    return (
        config.max_chars
        - len(heading_text)
        - 2
    )


def _prepare_oversized_units(
    *,
    heading_path: tuple[str, ...],
    units: Sequence[_ContentUnit],
    config: ChunkingConfig,
) -> _UnitPreparationResult:
    """Split oversized text units and preserve atomic units."""

    prepared_units: list[_ContentUnit] = []
    long_block_split_count = 0
    oversized_atomic_block_count = 0

    for unit in units:
        single_unit_draft = _ChunkDraft(
            heading_path=heading_path,
            units=(unit,),
        )

        if (
            len(
                _render_draft_index_text(
                    single_unit_draft
                )
            )
            <= config.max_chars
        ):
            prepared_units.append(unit)
            continue

        block_types = {
            block.block_type
            for block in unit.blocks
        }

        if block_types.issubset(
            _ATOMIC_BODY_TYPES
        ):
            prepared_units.append(unit)
            oversized_atomic_block_count += 1
            continue

        if not block_types.issubset(
            _SPLITTABLE_BODY_TYPES
        ):
            raise ValueError(
                "content unit contains incompatible "
                "block types"
            )

        content_budget = _content_char_budget(
            heading_path=heading_path,
            config=config,
        )

        fragments = _split_text_to_budget(
            unit.text,
            content_budget,
        )

        prepared_units.extend(
            _ContentUnit(
                text=fragment,
                blocks=unit.blocks,
            )
            for fragment in fragments
        )

        long_block_split_count += 1

    return _UnitPreparationResult(
        units=tuple(prepared_units),
        long_block_split_count=(
            long_block_split_count
        ),
        oversized_atomic_block_count=(
            oversized_atomic_block_count
        ),
    )


def _prepare_chunk_drafts(
    *,
    blocks: Sequence[NormalizedBlock],
    config: ChunkingConfig,
) -> _DraftPreparationResult:
    """Run normalized blocks through the draft pipeline."""

    groups = _group_body_blocks(blocks)

    drafts: list[_ChunkDraft] = []
    cross_page_join_count = 0
    long_block_split_count = 0
    oversized_atomic_block_count = 0

    for group in groups:
        merged_units, group_join_count = (
            _merge_cross_page_paragraphs(
                group
            )
        )

        preparation = _prepare_oversized_units(
            heading_path=group.heading_path,
            units=merged_units,
            config=config,
        )

        group_drafts = _pack_content_units(
            heading_path=group.heading_path,
            units=preparation.units,
            config=config,
        )

        drafts.extend(group_drafts)

        cross_page_join_count += (
            group_join_count
        )
        long_block_split_count += (
            preparation.long_block_split_count
        )
        oversized_atomic_block_count += (
            preparation.oversized_atomic_block_count
        )

    return _DraftPreparationResult(
        drafts=tuple(drafts),
        cross_page_join_count=(
            cross_page_join_count
        ),
        long_block_split_count=(
            long_block_split_count
        ),
        oversized_atomic_block_count=(
            oversized_atomic_block_count
        ),
    )
