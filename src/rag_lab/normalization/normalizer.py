from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_lab.contracts.blocks import (
    BlockType,
    NormalizedBlock,
)
from rag_lab.normalization.models import (
    NormalizationReport,
    NormalizationResult,
)


_CJK = "\u3400-\u4dbf\u4e00-\u9fff"
_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+)+)"
    r"(?:\s+|(?=[^\d.]))"
    r"(?P<title>.+)$"
)
_CHAPTER_MARKER = re.compile(
    r"^第\s*(?P<number>\d+)\s*章$"
)


@dataclass(frozen=True, slots=True)
class _RawBlock:
    source_ref: str
    original_index: int
    label: str
    text: str
    page_start: int
    page_end: int
    top_from_page: float
    left: float
    parent_ref: str | None


@dataclass(frozen=True, slots=True)
class _CandidateBlock:
    text: str
    block_type: BlockType
    heading_path: tuple[str, ...]
    page_start: int
    page_end: int


def compute_document_id(
    source_path: Path,
) -> str:
    digest = hashlib.sha256()

    with source_path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return f"sha256:{digest.hexdigest()}"


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()

    # Docling may preserve typesetting spaces inside outline numbers.
    while True:
        updated = re.sub(
            r"(?<=\d)\.\s+(?=\d)",
            ".",
            value,
        )

        if updated == value:
            break

        value = updated

    value = re.sub(
        rf"(?<=[{_CJK}]) +(?=[{_CJK}])",
        "",
        value,
    )
    value = re.sub(
        r" +(?=[，。；：？！、）】》])",
        "",
        value,
    )
    value = re.sub(
        r"(?<=[，。；：？！、]) +",
        "",
        value,
    )
    value = re.sub(
        r"(?<=[（【《]) +",
        "",
        value,
    )
    value = re.sub(
        r" +(?=[（【《])",
        "",
        value,
    )
    value = re.sub(
        rf"(?<=[）】》]) +(?=[{_CJK}])",
        "",
        value,
    )
    value = re.sub(
        r"(?<=[A-Za-z0-9])\s*/\s*"
        r"(?=[A-Za-z0-9])",
        "/",
        value,
    )
    value = re.sub(
        r"\s+([％%])",
        r"\1",
        value,
    )

    return value.strip()


def normalize_docling_document(
    *,
    docling_document: dict[str, Any],
    source_path: Path,
    normalization_version: str,
) -> NormalizationResult:
    resolved_source = source_path.resolve()

    if not resolved_source.is_file():
        raise FileNotFoundError(
            f"source file not found: {resolved_source}"
        )

    if not normalization_version.strip():
        raise ValueError(
            "normalization_version cannot be empty"
        )

    document_id = compute_document_id(
        resolved_source
    )
    source_path_text = resolved_source.as_posix()

    (
        raw_blocks,
        removed_furniture_count,
        source_pages,
        chapter_marker,
    ) = _extract_raw_blocks(docling_document)

    ordered_blocks, reordered_block_count = (
        _restore_single_column_order(raw_blocks)
    )

    (
        candidates,
        downgraded_heading_count,
        downgraded_heading_pages,
    ) = (
        _build_candidates(
            ordered_blocks,
            chapter_marker=chapter_marker,
        )
    )

    normalized_blocks = tuple(
        NormalizedBlock(
            block_id=_compute_block_id(
                document_id=document_id,
                candidate=candidate,
                ordinal=index,
                normalization_version=(
                    normalization_version
                ),
            ),
            document_id=document_id,
            text=candidate.text,
            block_type=(
                candidate.block_type.value
            ),
            heading_path=list(
                candidate.heading_path
            ),
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            ordinal=index,
            source_path=source_path_text,
            image_path=None,
            normalization_version=(
                normalization_version
            ),
        )
        for index, candidate in enumerate(
            candidates,
            start=1,
        )
    )

    _validate_blocks(
        normalized_blocks,
        source_pages=source_pages,
    )

    raw_count = len(raw_blocks)
    short_count = sum(
        len(normalize_text(block.text)) <= 3
        for block in raw_blocks
    )
    short_fragment_ratio = (
        short_count / raw_count
        if raw_count
        else 0.0
    )

    represented_pages = {
        page
        for block in normalized_blocks
        for page in range(
            block.page_start,
            block.page_end + 1,
        )
    }
    pages_requiring_review = tuple(
        sorted(
            {
                page
                for page in source_pages
                if page not in represented_pages
            }
            | set(downgraded_heading_pages)
        )
    )

    report = NormalizationReport(
        document_id=document_id,
        normalization_version=(
            normalization_version
        ),
        source_path=source_path_text,
        source_pages=source_pages,
        raw_block_count=raw_count,
        normalized_block_count=len(
            normalized_blocks
        ),
        removed_furniture_count=(
            removed_furniture_count
        ),
        reordered_block_count=(
            reordered_block_count
        ),
        downgraded_heading_count=(
            downgraded_heading_count
        ),
        short_fragment_ratio=(
            short_fragment_ratio
        ),
        pages_requiring_review=(
            pages_requiring_review
        ),
    )

    return NormalizationResult(
        blocks=normalized_blocks,
        report=report,
    )


def _compute_block_id(
    *,
    document_id: str,
    candidate: _CandidateBlock,
    ordinal: int,
    normalization_version: str,
) -> str:
    digest = hashlib.sha256()
    parts = (
        document_id,
        normalization_version,
        str(ordinal),
        str(candidate.page_start),
        str(candidate.page_end),
        candidate.block_type.value,
        "\n".join(candidate.heading_path),
        candidate.text,
    )

    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(
            len(encoded).to_bytes(
                8,
                byteorder="big",
            )
        )
        digest.update(encoded)

    return f"sha256:{digest.hexdigest()}"


def _extract_raw_blocks(
    document: dict[str, Any],
) -> tuple[
    list[_RawBlock],
    int,
    tuple[int, ...],
    str | None,
]:
    pages = document.get("pages")
    texts = document.get("texts")

    if not isinstance(pages, dict) or not pages:
        raise ValueError(
            "Docling JSON must contain pages"
        )

    if not isinstance(texts, list):
        raise ValueError(
            "Docling JSON must contain texts"
        )

    page_heights: dict[int, float] = {}

    for key, page in pages.items():
        if not isinstance(page, dict):
            raise ValueError(
                f"invalid page record: {key}"
            )

        page_number = int(
            page.get("page_no", key)
        )
        size = page.get("size")

        if not isinstance(size, dict):
            raise ValueError(
                f"page {page_number} has no size"
            )

        page_heights[page_number] = float(
            size["height"]
        )

    raw_blocks: list[_RawBlock] = []
    removed_furniture_count = 0
    chapter_markers: list[
        tuple[int, str]
    ] = []

    for index, item in enumerate(texts):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or "").strip()
        label = str(item.get("label") or "text")
        content_layer = str(
            item.get("content_layer") or "body"
        )
        provenance = item.get("prov")
        parent = item.get("parent")
        parent_ref = (
            str(parent.get("$ref"))
            if isinstance(parent, dict)
            and parent.get("$ref")
            else None
        )

        if (
            label in {
                "page_header",
                "page_footer",
            }
            or content_layer == "furniture"
        ):
            marker_match = _CHAPTER_MARKER.match(
                normalize_text(text)
            )

            if (
                marker_match
                and isinstance(provenance, list)
                and provenance
                and isinstance(provenance[0], dict)
                and "page_no" in provenance[0]
            ):
                marker_page = int(
                    provenance[0]["page_no"]
                )
                chapter_markers.append(
                    (
                        marker_page,
                        "第"
                        + marker_match.group(
                            "number"
                        )
                        + "章",
                    )
                )

            removed_furniture_count += 1
            continue

        if (
            parent_ref
            and parent_ref.startswith(
                "#/pictures/"
            )
            and label != "caption"
        ):
            continue

        if not text:
            continue

        if (
            not isinstance(provenance, list)
            or not provenance
        ):
            raise ValueError(
                f"text item {index} has no provenance"
            )

        page_numbers = [
            int(prov["page_no"])
            for prov in provenance
            if isinstance(prov, dict)
            and "page_no" in prov
        ]

        if not page_numbers:
            raise ValueError(
                f"text item {index} has no page"
            )

        first_page = min(page_numbers)
        first_provenance = next(
            prov
            for prov in provenance
            if int(prov["page_no"])
            == first_page
        )
        bbox = first_provenance.get("bbox")

        if not isinstance(bbox, dict):
            raise ValueError(
                f"text item {index} has no bbox"
            )

        top = float(bbox["t"])
        left = float(bbox["l"])
        origin = str(
            bbox.get(
                "coord_origin",
                "TOPLEFT",
            )
        )

        if origin == "BOTTOMLEFT":
            top_from_page = (
                page_heights[first_page] - top
            )
        elif origin == "TOPLEFT":
            top_from_page = top
        else:
            raise ValueError(
                f"unsupported coordinate origin: "
                f"{origin}"
            )

        raw_blocks.append(
            _RawBlock(
                source_ref=str(
                    item.get(
                        "self_ref",
                        f"#/texts/{index}",
                    )
                ),
                original_index=index,
                label=label,
                text=text,
                page_start=first_page,
                page_end=max(page_numbers),
                top_from_page=top_from_page,
                left=left,
                parent_ref=parent_ref,
            )
        )

    if not raw_blocks:
        raise ValueError(
            "Docling JSON contains no usable text"
        )

    return (
        raw_blocks,
        removed_furniture_count,
        tuple(sorted(page_heights)),
        (
            min(chapter_markers)[1]
            if chapter_markers
            else None
        ),
    )


def _restore_single_column_order(
    blocks: list[_RawBlock],
) -> tuple[list[_RawBlock], int]:
    ordered = sorted(
        blocks,
        key=lambda block: (
            block.page_start,
            block.top_from_page,
            block.left,
            block.original_index,
        ),
    )

    original_positions = {
        block.source_ref: index
        for index, block in enumerate(blocks)
    }
    reordered_count = sum(
        original_positions[block.source_ref]
        != index
        for index, block in enumerate(ordered)
    )

    return ordered, reordered_count


def _build_candidates(
    blocks: list[_RawBlock],
    *,
    chapter_marker: str | None,
) -> tuple[
    list[_CandidateBlock],
    int,
    tuple[int, ...],
]:
    candidates: list[_CandidateBlock] = []
    heading_stack: list[str] = []
    downgraded_heading_count = 0
    downgraded_heading_pages: set[int] = (
        set()
    )
    title_consumed = False

    for block in blocks:
        text = normalize_text(block.text)

        if not text:
            continue

        heading_match = (
            _NUMBERED_HEADING.match(text)
        )

        if (
            block.label == "section_header"
            and not title_consumed
            and not heading_match
        ):
            title = " ".join(
                part
                for part in (
                    chapter_marker,
                    text,
                )
                if part
            )
            heading_stack = [title]
            title_consumed = True
            candidates.append(
                _CandidateBlock(
                    text=title,
                    block_type=(
                        BlockType.DOCUMENT_TITLE
                    ),
                    heading_path=(title,),
                    page_start=block.page_start,
                    page_end=block.page_end,
                )
            )
            continue

        if (
            chapter_marker
            and not title_consumed
        ):
            # A chapter marker may appear in furniture above the
            # actual chapter title. Text between those two items is
            # running furniture, not body content.
            continue

        if (
            block.label == "section_header"
            and heading_match
        ):
            number = heading_match.group(
                "number"
            )
            title_text = heading_match.group(
                "title"
            ).strip()
            heading = (
                f"{number} {title_text}"
            )
            level = number.count(".") + 1

            if not heading_stack:
                heading_stack = [
                    chapter_marker
                    or "Document"
                ]

            heading_stack = (
                heading_stack[:level - 1]
                + [heading]
            )
            candidates.append(
                _CandidateBlock(
                    text=heading,
                    block_type=(
                        BlockType.SECTION_HEADING
                    ),
                    heading_path=tuple(
                        heading_stack
                    ),
                    page_start=block.page_start,
                    page_end=block.page_end,
                )
            )
            continue

        if block.label == "section_header":
            downgraded_heading_count += 1
            downgraded_heading_pages.add(
                block.page_start
            )

        if not heading_stack:
            fallback_title = (
                chapter_marker or "Document"
            )
            heading_stack = [fallback_title]

        block_type = _map_block_type(
            block.label
        )
        candidates.append(
            _CandidateBlock(
                text=text,
                block_type=block_type,
                heading_path=tuple(
                    heading_stack
                ),
                page_start=block.page_start,
                page_end=block.page_end,
            )
        )

    return (
        candidates,
        downgraded_heading_count,
        tuple(sorted(downgraded_heading_pages)),
    )


def _map_block_type(
    label: str,
) -> BlockType:
    mapping = {
        "caption": BlockType.FIGURE_CAPTION,
        "list_item": BlockType.LIST_ITEM,
        "table": BlockType.TABLE,
        "code": BlockType.CODE,
        "formula": BlockType.EQUATION,
    }

    return mapping.get(
        label,
        BlockType.PARAGRAPH,
    )


def _validate_blocks(
    blocks: tuple[NormalizedBlock, ...],
    *,
    source_pages: tuple[int, ...],
) -> None:
    if not blocks:
        raise ValueError(
            "normalization produced no blocks"
        )

    expected_ordinals = list(
        range(1, len(blocks) + 1)
    )
    actual_ordinals = [
        block.ordinal
        for block in blocks
    ]

    if actual_ordinals != expected_ordinals:
        raise ValueError(
            "block ordinals must be contiguous"
        )

    valid_pages = set(source_pages)

    for block in blocks:
        if (
            block.page_start not in valid_pages
            or block.page_end not in valid_pages
        ):
            raise ValueError(
                f"block {block.ordinal} references "
                "a page outside the source range"
            )
