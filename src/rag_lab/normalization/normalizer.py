from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
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
from rag_lab.normalization.corrections import (
    Correction,
    CorrectionOverlay,
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
_ORPHAN_PUNCTUATION = frozenset(
    {
        "。",
        "，",
        "；",
        "：",
        "、",
        ".",
        ",",
        ";",
        ":",
    }
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
    image_path: str | None


@dataclass(frozen=True, slots=True)
class _CandidateBlock:
    source_ref: str
    text: str
    block_type: BlockType
    heading_path: tuple[str, ...]
    page_start: int
    page_end: int
    image_path: str | None


@dataclass(frozen=True, slots=True)
class _ChapterMarker:
    text: str
    page: int
    top_from_page: float
    left: float
    original_index: int


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
    artifact_directory: Path | None = None,
    correction_overlay: CorrectionOverlay | None = None,
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
        chapter_markers,
    ) = _extract_raw_blocks(
        docling_document,
        artifact_directory=artifact_directory,
    )
    formula_marker_pages = _extract_formula_marker_pages(
        docling_document
    )

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
            chapter_markers=chapter_markers,
        )
    )

    correction_summary: dict[str, int] | None = None
    if correction_overlay is not None:
        candidates, correction_summary = (
            _apply_correction_overlay(
                candidates,
                overlay=correction_overlay,
                document_id=document_id,
                formula_marker_pages=(
                    formula_marker_pages
                ),
            )
        )

    (
        candidates,
        merged_orphan_punctuation_count,
        non_indexed_orphan_punctuation_count,
    ) = _handle_orphan_punctuation(candidates)

    normalized_blocks = tuple(
        NormalizedBlock(
            block_id=_compute_block_id(
                document_id=document_id,
                candidate=candidate,
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
            image_path=candidate.image_path,
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
        merged_orphan_punctuation_count=(
            merged_orphan_punctuation_count
        ),
        non_indexed_orphan_punctuation_count=(
            non_indexed_orphan_punctuation_count
        ),
        short_fragment_ratio=(
            short_fragment_ratio
        ),
        pages_requiring_review=(
            pages_requiring_review
        ),
        correction_summary=correction_summary,
        correction_overlay=(
            _correction_overlay_report(
                overlay=correction_overlay,
                candidates=candidates,
                blocks=normalized_blocks,
            )
            if correction_overlay is not None
            else None
        ),
    )

    return NormalizationResult(
        blocks=normalized_blocks,
        report=report,
    )


def _correction_overlay_report(
    *,
    overlay: CorrectionOverlay,
    candidates: list[_CandidateBlock],
    blocks: tuple[NormalizedBlock, ...],
) -> dict[str, object]:
    block_by_source_ref = {
        candidate.source_ref: block
        for candidate, block in zip(candidates, blocks)
    }
    return {
        "schema_version": overlay.schema_version,
        "sha256": overlay.sha256,
        "source_path": overlay.source_path,
        "applied_correction_ids": [
            correction.correction_id for correction in overlay.corrections
        ],
        "formula_restorations": [
            {
                "correction_id": correction.correction_id,
                "marker_line": correction.marker_line,
                "marker_page": correction.page,
                "marker_source_ref": (
                    correction.marker_source_ref
                ),
                "anchor_source_ref": correction.source_refs[0],
                "equation_block_id": block_by_source_ref[
                    f"overlay:{correction.correction_id}"
                ].block_id,
            }
            for correction in overlay.corrections
            if correction.operation == "insert_equation"
        ],
    }


def _compute_block_id(
    *,
    document_id: str,
    candidate: _CandidateBlock,
    normalization_version: str,
) -> str:
    digest = hashlib.sha256()
    parts = (
        document_id,
        normalization_version,
        candidate.source_ref,
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


def _apply_correction_overlay(
    candidates: list[_CandidateBlock],
    *,
    overlay: CorrectionOverlay,
    document_id: str,
    formula_marker_pages: dict[str, int],
) -> tuple[list[_CandidateBlock], dict[str, int]]:
    """Apply a fixed set of source-anchored corrections before ID creation."""

    if overlay.document_id != document_id:
        raise ValueError(
            "correction overlay document_id does not match source"
        )

    corrected = list(candidates)
    summary: dict[str, int] = {}
    original_insert_context = _insert_anchor_context(candidates)

    for correction in overlay.corrections:
        _validate_formula_marker_reference(
            correction=correction,
            formula_marker_pages=formula_marker_pages,
        )
        locations = _correction_locations(
            corrected,
            correction=correction,
        )
        _validate_correction_anchors(
            corrected,
            correction=correction,
            locations=locations,
            original_insert_context=(
                original_insert_context
            ),
        )

        if correction.operation == "replace_text":
            _replace_correction_text(
                corrected,
                correction=correction,
                location=locations[0],
            )
        elif correction.operation == "merge_text":
            corrected = _merge_correction_text(
                corrected,
                correction=correction,
                locations=locations,
            )
        elif correction.operation == "reorder_blocks":
            _reorder_correction_blocks(
                corrected,
                correction=correction,
                locations=locations,
            )
        elif correction.operation == "reclassify_block":
            _reclassify_correction_block(
                corrected,
                correction=correction,
                location=locations[0],
            )
        elif correction.operation == "exclude_from_index":
            _exclude_correction_block(
                corrected,
                location=locations[0],
            )
        elif correction.operation == "insert_equation":
            corrected = _insert_equation_correction(
                corrected,
                correction=correction,
                location=locations[0],
            )
        else:
            raise AssertionError(
                f"unsupported correction operation: "
                f"{correction.operation}"
            )

        summary[correction.operation] = (
            summary.get(correction.operation, 0) + 1
        )

    summary["formula_restoration_count"] = summary.get(
        "insert_equation",
        0,
    )
    return corrected, summary


def _insert_anchor_context(
    candidates: list[_CandidateBlock],
) -> dict[str, tuple[str, str]]:
    return {
        candidate.source_ref: (
            candidates[index - 1].text,
            candidates[index + 1].text,
        )
        for index, candidate in enumerate(candidates)
        if 0 < index < len(candidates) - 1
    }


def _validate_formula_marker_reference(
    *,
    correction: Correction,
    formula_marker_pages: dict[str, int],
) -> None:
    if correction.operation != "insert_equation":
        return

    marker_source_ref = correction.marker_source_ref
    if marker_source_ref is None:
        # Direct Python callers that construct the legacy dataclass still
        # exercise the normalizer tests. JSON overlays must supply the field.
        return

    marker_page = formula_marker_pages.get(marker_source_ref)
    if marker_page is None:
        raise ValueError(
            f"correction {correction.correction_id} references "
            "missing formula marker source_ref: "
            f"{marker_source_ref}"
        )

    if marker_page != correction.page:
        raise ValueError(
            f"correction {correction.correction_id} marker source_ref "
            "page does not match"
        )


def _correction_locations(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
) -> tuple[int, ...]:
    by_source_ref = {
        candidate.source_ref: index
        for index, candidate in enumerate(candidates)
    }

    try:
        locations = tuple(
            by_source_ref[source_ref]
            for source_ref in correction.source_refs
        )
    except KeyError as error:
        raise ValueError(
            f"correction {correction.correction_id} references "
            f"missing source_ref: {error.args[0]}"
        ) from error

    if (
        correction.operation != "insert_equation"
        and any(
            candidates[location].page_start != correction.page
            for location in locations
        )
    ):
        raise ValueError(
            f"correction {correction.correction_id} page does not "
            "match its source_ref"
        )

    return locations


def _validate_correction_anchors(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    locations: tuple[int, ...],
    original_insert_context: dict[str, tuple[str, str]],
) -> None:
    if correction.operation == "insert_equation":
        anchor_context = original_insert_context.get(
            correction.source_refs[0]
        )
        if anchor_context is None:
            raise ValueError(
                f"correction {correction.correction_id} has no "
                "original insertion-anchor context"
            )
        before, after = anchor_context
    else:
        first_location = min(locations)
        last_location = max(locations)

        if first_location == 0:
            raise ValueError(
                f"correction {correction.correction_id} has no "
                "before-text anchor"
            )

        if last_location == len(candidates) - 1:
            raise ValueError(
                f"correction {correction.correction_id} has no "
                "after-text anchor"
            )

        before = candidates[first_location - 1].text
        after = candidates[last_location + 1].text
    if correction.before_text not in before:
        raise ValueError(
            f"correction {correction.correction_id} before_text "
            "anchor does not match"
        )

    if correction.after_text not in after:
        raise ValueError(
            f"correction {correction.correction_id} after_text "
            "anchor does not match"
        )

def _replace_correction_text(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    location: int,
) -> None:
    replacement = correction.replacement
    if replacement is None:
        raise AssertionError("replace_text requires replacement")

    candidate = candidates[location]
    if correction.find_text is None:
        corrected_text = replacement
    else:
        if candidate.text.count(correction.find_text) != 1:
            raise ValueError(
                f"correction {correction.correction_id} find_text "
                "must occur exactly once"
            )
        corrected_text = candidate.text.replace(
            correction.find_text,
            replacement,
        )

    candidates[location] = replace(
        candidate,
        text=corrected_text,
    )


def _merge_correction_text(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    locations: tuple[int, ...],
) -> list[_CandidateBlock]:
    replacement = correction.replacement
    if replacement is None:
        raise AssertionError("merge_text requires replacement")

    ordered_locations = sorted(locations)
    selected = [
        candidates[location]
        for location in ordered_locations
    ]
    first = selected[0]
    block_type = (
        BlockType(correction.block_type)
        if correction.block_type is not None
        else first.block_type
    )
    merged = _CandidateBlock(
        source_ref=(
            "overlay:"
            f"{correction.correction_id}"
        ),
        text=replacement,
        block_type=block_type,
        heading_path=first.heading_path,
        page_start=min(
            candidate.page_start
            for candidate in selected
        ),
        page_end=max(
            candidate.page_end
            for candidate in selected
        ),
        image_path=None,
    )
    selected_locations = set(ordered_locations)
    insertion_location = ordered_locations[0]
    merged_candidates: list[_CandidateBlock] = []

    for location, candidate in enumerate(candidates):
        if location == insertion_location:
            merged_candidates.append(merged)

        if location not in selected_locations:
            merged_candidates.append(candidate)

    return merged_candidates


def _reorder_correction_blocks(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    locations: tuple[int, ...],
) -> None:
    selected = [
        candidates[location]
        for location in locations
    ]
    for location, candidate in zip(
        sorted(locations),
        selected,
    ):
        candidates[location] = candidate


def _reclassify_correction_block(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    location: int,
) -> None:
    if correction.block_type is None:
        raise AssertionError(
            "reclassify_block requires block_type"
        )

    block_type = BlockType(correction.block_type)
    candidate = candidates[location]
    if block_type != BlockType.SECTION_HEADING:
        candidates[location] = replace(
            candidate,
            block_type=block_type,
        )
        return

    old_heading_path = candidate.heading_path
    new_heading_path = old_heading_path + (
        candidate.text,
    )
    candidates[location] = replace(
        candidate,
        block_type=block_type,
        heading_path=new_heading_path,
    )

    for index in range(location + 1, len(candidates)):
        following = candidates[index]
        if following.heading_path[: len(old_heading_path)] != (
            old_heading_path
        ):
            continue

        candidates[index] = replace(
            following,
            heading_path=(
                new_heading_path
                + following.heading_path[
                    len(old_heading_path) :
                ]
            ),
        )


def _exclude_correction_block(
    candidates: list[_CandidateBlock],
    *,
    location: int,
) -> None:
    candidates[location] = replace(
        candidates[location],
        block_type=BlockType.FIGURE_LABEL,
    )


def _insert_equation_correction(
    candidates: list[_CandidateBlock],
    *,
    correction: Correction,
    location: int,
) -> list[_CandidateBlock]:
    replacement = correction.replacement
    if replacement is None:
        raise AssertionError("insert_equation requires replacement")

    anchor = candidates[location]
    equation = _CandidateBlock(
        source_ref=(
            "overlay:"
            f"{correction.correction_id}"
        ),
        text=replacement,
        block_type=BlockType.EQUATION,
        heading_path=anchor.heading_path,
        page_start=correction.page,
        page_end=correction.page,
        image_path=None,
    )
    return [
        *candidates[: location + 1],
        equation,
        *candidates[location + 1 :],
    ]


def _extract_caption_image_paths(
    document: dict[str, Any],
    *,
    artifact_directory: Path | None,
) -> dict[str, str]:
    pictures = document.get("pictures")

    if not isinstance(pictures, list):
        return {}

    image_paths: dict[str, str] = {}

    for picture in pictures:
        if not isinstance(picture, dict):
            continue

        image = picture.get("image")
        captions = picture.get("captions")

        if (
            not isinstance(image, dict)
            or not isinstance(captions, list)
        ):
            continue

        uri = image.get("uri")

        if not isinstance(uri, str) or not uri:
            continue

        relative_path = _relative_image_path(
            uri,
            artifact_directory=(
                artifact_directory
            ),
        )

        if relative_path is None:
            continue

        for caption in captions:
            if not isinstance(caption, dict):
                continue

            caption_ref = caption.get("$ref")

            if not isinstance(caption_ref, str):
                continue

            existing = image_paths.get(
                caption_ref
            )

            if (
                existing is not None
                and existing != relative_path
            ):
                raise ValueError(
                    f"caption {caption_ref} refers "
                    "to multiple images"
                )

            image_paths[caption_ref] = (
                relative_path
            )

    return image_paths


def _relative_image_path(
    uri: str,
    *,
    artifact_directory: Path | None,
) -> str | None:
    image_path = Path(uri)
    relative_path: Path | None = None
    artifact_root = (
        artifact_directory.resolve()
        if artifact_directory is not None
        else None
    )

    if not image_path.is_absolute():
        relative_path = image_path
    elif artifact_root is not None:
        try:
            relative_path = (
                image_path.resolve().relative_to(
                    artifact_root
                )
            )
        except ValueError:
            relative_path = None

    if relative_path is None:
        lower_parts = [
            part.lower()
            for part in image_path.parts
        ]

        if "assets" in lower_parts:
            asset_index = len(lower_parts) - 1 - (
                lower_parts[::-1].index(
                    "assets"
                )
            )
            relative_path = Path(
                *image_path.parts[asset_index:]
            )

    if (
        relative_path is None
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        return None

    if (
        artifact_root is not None
        and not (
            artifact_root / relative_path
        ).is_file()
    ):
        return None

    return relative_path.as_posix()


def _extract_raw_blocks(
    document: dict[str, Any],
    *,
    artifact_directory: Path | None,
) -> tuple[
    list[_RawBlock],
    int,
    tuple[int, ...],
    tuple[_ChapterMarker, ...],
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
    chapter_markers: list[_ChapterMarker] = []
    caption_image_paths = (
        _extract_caption_image_paths(
            document,
            artifact_directory=(
                artifact_directory
            ),
        )
    )

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
                and isinstance(
                    provenance[0].get("bbox"),
                    dict,
                )
            ):
                marker_page = int(
                    provenance[0]["page_no"]
                )
                marker_bbox = provenance[0][
                    "bbox"
                ]
                marker_top = float(
                    marker_bbox["t"]
                )
                marker_origin = str(
                    marker_bbox.get(
                        "coord_origin",
                        "TOPLEFT",
                    )
                )

                if marker_origin == "BOTTOMLEFT":
                    marker_top_from_page = (
                        page_heights[marker_page]
                        - marker_top
                    )
                elif marker_origin == "TOPLEFT":
                    marker_top_from_page = (
                        marker_top
                    )
                else:
                    raise ValueError(
                        "unsupported coordinate "
                        f"origin: {marker_origin}"
                    )

                chapter_markers.append(
                    _ChapterMarker(
                        text=(
                            "第"
                            + marker_match.group(
                                "number"
                            )
                            + "章"
                        ),
                        page=marker_page,
                        top_from_page=(
                            marker_top_from_page
                        ),
                        left=float(
                            marker_bbox["l"]
                        ),
                        original_index=index,
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
                image_path=(
                    caption_image_paths.get(
                        str(
                            item.get(
                                "self_ref",
                                f"#/texts/{index}",
                            )
                        )
                    )
                ),
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
        tuple(chapter_markers),
    )


def _extract_formula_marker_pages(
    document: dict[str, Any],
) -> dict[str, int]:
    texts = document.get("texts")
    if not isinstance(texts, list):
        raise ValueError("Docling JSON must contain texts")

    marker_pages: dict[str, int] = {}
    for index, item in enumerate(texts):
        if (
            not isinstance(item, dict)
            or item.get("label") != "formula"
            or str(item.get("text") or "").strip()
        ):
            continue

        provenance = item.get("prov")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError(
                f"formula marker {index} has no provenance"
            )

        first = provenance[0]
        if not isinstance(first, dict) or "page_no" not in first:
            raise ValueError(
                f"formula marker {index} has no page"
            )

        source_ref = str(
            item.get("self_ref", f"#/texts/{index}")
        )
        if source_ref in marker_pages:
            raise ValueError(
                f"duplicate formula marker source_ref: {source_ref}"
            )
        marker_pages[source_ref] = int(first["page_no"])

    return marker_pages


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
    chapter_markers: tuple[
        _ChapterMarker,
        ...,
    ],
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
    document_title_consumed = False
    pending_chapter_marker: str | None = (
        None
    )
    events: list[
        tuple[
            tuple[int, float, float, int, int],
            _RawBlock | _ChapterMarker,
        ]
    ] = []

    for marker in chapter_markers:
        events.append(
            (
                (
                    marker.page,
                    marker.top_from_page,
                    marker.left,
                    marker.original_index,
                    0,
                ),
                marker,
            )
        )

    for block in blocks:
        events.append(
            (
                (
                    block.page_start,
                    block.top_from_page,
                    block.left,
                    block.original_index,
                    1,
                ),
                block,
            )
        )

    active_chapter_marker: str | None = None
    for _, event in sorted(events):
        if isinstance(event, _ChapterMarker):
            if event.text == active_chapter_marker:
                continue
            pending_chapter_marker = event.text
            heading_stack = []
            active_chapter_marker = event.text
            continue

        block = event
        text = normalize_text(block.text)

        if not text:
            continue

        heading_match = (
            _NUMBERED_HEADING.match(text)
        )

        if (
            block.label == "section_header"
            and not heading_match
            and (
                pending_chapter_marker
                or not document_title_consumed
            )
        ):
            title = " ".join(
                part
                for part in (
                    pending_chapter_marker,
                    text,
                )
                if part
            )
            heading_stack = [title]
            block_type = (
                BlockType.DOCUMENT_TITLE
                if not document_title_consumed
                else BlockType.SECTION_HEADING
            )
            document_title_consumed = True
            pending_chapter_marker = None
            candidates.append(
                _CandidateBlock(
                    source_ref=block.source_ref,
                    text=title,
                    block_type=block_type,
                    heading_path=(title,),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    image_path=block.image_path,
                )
            )
            continue

        if pending_chapter_marker:
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
                    "Document"
                ]

            heading_stack = (
                heading_stack[:level - 1]
                + [heading]
            )
            candidates.append(
                _CandidateBlock(
                    source_ref=block.source_ref,
                    text=heading,
                    block_type=(
                        BlockType.SECTION_HEADING
                    ),
                    heading_path=tuple(
                        heading_stack
                    ),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    image_path=block.image_path,
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
                pending_chapter_marker or "Document"
            )
            heading_stack = [fallback_title]

        block_type = _map_block_type(
            block.label
        )
        candidates.append(
            _CandidateBlock(
                source_ref=block.source_ref,
                text=text,
                block_type=block_type,
                heading_path=tuple(
                    heading_stack
                ),
                page_start=block.page_start,
                page_end=block.page_end,
                image_path=block.image_path,
            )
        )

    return (
        candidates,
        downgraded_heading_count,
        tuple(sorted(downgraded_heading_pages)),
    )


def _handle_orphan_punctuation(
    candidates: list[_CandidateBlock],
) -> tuple[list[_CandidateBlock], int, int]:
    """Handle punctuation-only body items without discarding source records.

    Docling sometimes emits a sentence-ending mark as its own body item.
    When the preceding same-page body item does not already end in
    punctuation, the mark is appended to it. Otherwise, preserving its
    intended syntactic relationship would require a source-specific reviewed
    correction. The standalone record is retained as non-indexable instead of
    being silently deleted or emitted into retrieval chunks.
    """

    merged = list(candidates)
    merged_count = 0
    non_indexed_count = 0
    body_types = {
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
    }

    for index, candidate in enumerate(merged):
        if (
            candidate.block_type not in body_types
            or candidate.text not in _ORPHAN_PUNCTUATION
            or index == 0
        ):
            continue

        previous = merged[index - 1]
        if (
            previous.block_type in body_types
            and previous.page_start == candidate.page_start
            and previous.page_end == candidate.page_end
            and previous.image_path is None
            and candidate.image_path is None
            and previous.text[-1] not in _ORPHAN_PUNCTUATION
        ):
            merged[index - 1] = replace(
                previous,
                text=previous.text + candidate.text,
            )
            merged_count += 1
        else:
            non_indexed_count += 1

        merged[index] = replace(
            candidate,
            block_type=BlockType.FIGURE_LABEL,
        )

    return merged, merged_count, non_indexed_count


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
