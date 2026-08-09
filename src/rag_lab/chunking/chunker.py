from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.chunking.models import (
    ChunkingConfig,
    ChunkingReport,
    ChunkingResult,
)

from rag_lab.contracts import (
    BlockType,
    KnowledgeChunk,
    NormalizedBlock,
)

_CONTROL_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        BlockType.DOCUMENT_TITLE.value,
        BlockType.SECTION_HEADING.value,
    }
)

_NON_INDEXABLE_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        BlockType.FIGURE_LABEL.value,
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
    overlap_unit_count: int = 0


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
    overlapped_chunk_count: int
    overlap_char_count: int


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
        if block.block_type in _NON_INDEXABLE_BLOCK_TYPES:
            continue

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


def _render_units_text(
    units: Sequence[_ContentUnit],
) -> str:
    """Render content units without heading context."""

    return "\n\n".join(
        unit.text
        for unit in units
    )


def _complete_sentence_suffix(
    unit: _ContentUnit,
    max_chars: int,
) -> _ContentUnit | None:
    """Return the longest complete-sentence suffix that fits."""

    if max_chars < 1:
        return None

    block_types = {
        block.block_type
        for block in unit.blocks
    }

    if not block_types.issubset(
        _SPLITTABLE_BODY_TYPES
    ):
        return None

    text = unit.text.strip()
    boundaries = _sentence_boundary_positions(text)

    if not boundaries or boundaries[-1] != len(text):
        return None

    sentence_starts = (0,) + boundaries[:-1]

    for start in sentence_starts:
        suffix = text[start:].strip()

        if suffix and len(suffix) <= max_chars:
            return _ContentUnit(
                text=suffix,
                blocks=unit.blocks,
            )

    return None


def _select_overlap_units(
    *,
    heading_path: tuple[str, ...],
    source_units: Sequence[_ContentUnit],
    next_unit: _ContentUnit,
    config: ChunkingConfig,
) -> tuple[_ContentUnit, ...]:
    """Select a provenance-preserving suffix for the next draft."""

    if config.overlap_chars == 0 or not source_units:
        return ()

    next_draft = _ChunkDraft(
        heading_path=heading_path,
        units=(next_unit,),
    )
    next_length = len(
        _render_draft_index_text(next_draft)
    )

    if next_length > config.max_chars:
        return ()

    available_chars = min(
        config.overlap_chars,
        config.max_chars - next_length - 2,
    )

    if available_chars < 1:
        return ()

    selected: list[_ContentUnit] = []

    for unit in reversed(source_units):
        candidate = [unit, *selected]

        if len(
            _render_units_text(candidate)
        ) <= available_chars:
            selected.insert(0, unit)
            continue

        separator_chars = 2 if selected else 0
        suffix_budget = (
            available_chars
            - len(_render_units_text(selected))
            - separator_chars
        )
        suffix = _complete_sentence_suffix(
            unit,
            suffix_budget,
        )

        if suffix is not None:
            selected.insert(0, suffix)

        break

    overlap = tuple(selected)

    if not overlap:
        return ()

    candidate_draft = _ChunkDraft(
        heading_path=heading_path,
        units=overlap + (next_unit,),
        overlap_unit_count=len(overlap),
    )

    if (
        len(_render_draft_index_text(candidate_draft))
        > config.max_chars
    ):
        raise AssertionError(
            "selected overlap exceeds max_chars"
        )

    return overlap


def _draft_overlap_char_count(
    draft: _ChunkDraft,
) -> int:
    """Return repeated content characters in one draft."""

    if draft.overlap_unit_count == 0:
        return 0

    return len(
        _render_units_text(
            draft.units[
                :draft.overlap_unit_count
            ]
        )
    )


def _pack_content_units(
    *,
    heading_path: tuple[str, ...],
    units: Sequence[_ContentUnit],
    config: ChunkingConfig,
) -> tuple[_ChunkDraft, ...]:
    """Greedily pack ordered units under max_chars."""

    drafts: list[_ChunkDraft] = []
    current_units: list[_ContentUnit] = []
    current_overlap_unit_count = 0

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
                    overlap_unit_count=(
                        current_overlap_unit_count
                    ),
                )
            )

            primary_units = current_units[
                current_overlap_unit_count:
            ]
            overlap_units = _select_overlap_units(
                heading_path=heading_path,
                source_units=primary_units,
                next_unit=unit,
                config=config,
            )
            current_units = [
                *overlap_units,
                unit,
            ]
            current_overlap_unit_count = len(
                overlap_units
            )
            continue

        current_units.append(unit)

    if current_units:
        drafts.append(
            _ChunkDraft(
                heading_path=heading_path,
                units=tuple(current_units),
                overlap_unit_count=(
                    current_overlap_unit_count
                ),
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
    overlapped_chunk_count = 0
    overlap_char_count = 0

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
        overlapped_chunk_count += sum(
            draft.overlap_unit_count > 0
            for draft in group_drafts
        )
        overlap_char_count += sum(
            _draft_overlap_char_count(draft)
            for draft in group_drafts
        )

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
        overlapped_chunk_count=(
            overlapped_chunk_count
        ),
        overlap_char_count=overlap_char_count,
    )

def _hash_text(text: str) -> str:
    """Return a versioned SHA-256 text hash."""

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return f"sha256:{digest}"

def _ordered_draft_blocks(
    draft: _ChunkDraft,
) -> tuple[NormalizedBlock, ...]:
    """Return source blocks in first-appearance order."""

    ordered_blocks: list[NormalizedBlock] = []
    seen_block_ids: set[str] = set()

    for unit in draft.units:
        for block in unit.blocks:
            if block.block_id in seen_block_ids:
                continue

            seen_block_ids.add(block.block_id)
            ordered_blocks.append(block)

    if not ordered_blocks:
        raise ValueError(
            "chunk draft must contain source blocks"
        )

    return tuple(ordered_blocks)

def _build_chunk_id(
    *,
    document_id: str,
    heading_path: tuple[str, ...],
    block_ids: tuple[str, ...],
    content_hash: str,
    chunking_version: str,
) -> str:
    """Build a deterministic provenance-aware chunk ID."""

    identity = {
        "document_id": document_id,
        "heading_path": list(heading_path),
        "block_ids": list(block_ids),
        "content_hash": content_hash,
        "chunking_version": chunking_version,
    }

    canonical_identity = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return _hash_text(canonical_identity)

def _draft_to_knowledge_chunk(
    *,
    draft: _ChunkDraft,
    ordinal: int,
    config: ChunkingConfig,
) -> KnowledgeChunk:
    """Convert one internal draft to the public contract."""

    source_blocks = _ordered_draft_blocks(
        draft
    )

    document_ids = {
        block.document_id
        for block in source_blocks
    }
    source_paths = {
        block.source_path
        for block in source_blocks
    }
    normalization_versions = {
        block.normalization_version
        for block in source_blocks
    }

    if len(document_ids) != 1:
        raise ValueError(
            "chunk draft must have one document_id"
        )

    if len(source_paths) != 1:
        raise ValueError(
            "chunk draft must have one source_path"
        )

    if len(normalization_versions) != 1:
        raise ValueError(
            "chunk draft must have one "
            "normalization_version"
        )

    document_id = next(iter(document_ids))
    source_path = next(iter(source_paths))
    normalization_version = next(
        iter(normalization_versions)
    )

    content = _render_draft_content(draft)
    index_text = _render_draft_index_text(
        draft
    )
    content_hash = _hash_text(content)

    block_ids = tuple(
        block.block_id
        for block in source_blocks
    )

    chunk_id = _build_chunk_id(
        document_id=document_id,
        heading_path=draft.heading_path,
        block_ids=block_ids,
        content_hash=content_hash,
        chunking_version=(
            config.chunking_version
        ),
    )

    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        index_text=index_text,
        heading_path=list(
            draft.heading_path
        ),
        page_start=min(
            block.page_start
            for block in source_blocks
        ),
        page_end=max(
            block.page_end
            for block in source_blocks
        ),
        ordinal=ordinal,
        block_ids=list(block_ids),
        source_path=source_path,
        content_hash=content_hash,
        normalization_version=(
            normalization_version
        ),
        chunking_version=(
            config.chunking_version
        ),
    )

def chunk_normalized_blocks(
    *,
    blocks: Sequence[NormalizedBlock],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Convert normalized blocks into knowledge chunks."""

    resolved_config = (
        config
        if config is not None
        else ChunkingConfig()
    )

    preparation = _prepare_chunk_drafts(
        blocks=blocks,
        config=resolved_config,
    )

    # preparation 成功意味着输入非空且属于同一文档。
    document_id = blocks[0].document_id

    chunks = [
        _draft_to_knowledge_chunk(
            draft=draft,
            ordinal=ordinal,
            config=resolved_config,
        )
        for ordinal, draft in enumerate(
            preparation.drafts,
            start=1,
        )
    ]

    report = ChunkingReport(
        document_id=document_id,
        input_block_count=len(blocks),
        output_chunk_count=len(chunks),
        cross_page_join_count=(
            preparation.cross_page_join_count
        ),
        long_block_split_count=(
            preparation.long_block_split_count
        ),
        oversized_atomic_block_count=(
            preparation.oversized_atomic_block_count
        ),
        overlapped_chunk_count=(
            preparation.overlapped_chunk_count
        ),
        overlap_char_count=(
            preparation.overlap_char_count
        ),
    )

    return ChunkingResult(
        chunks=chunks,
        report=report,
    )
