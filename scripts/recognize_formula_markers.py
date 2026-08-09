"""Recognize only audited formula-marker pages from an existing conversion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_lab.ingestion import read_book_manifest
from rag_lab.normalization.normalizer import (
    _build_candidates,
    _extract_raw_blocks,
    _restore_single_column_order,
)


@dataclass(frozen=True)
class FormulaMarker:
    section_id: str
    page: int
    marker_line: int
    marker_source_ref: str
    anchor_source_ref: str
    before_text: str
    after_text: str
    bbox: tuple[float, float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract formula-not-decoded regions from an existing Docling "
            "conversion using the original PDF text layer."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--conversion-root",
        type=Path,
        required=True,
        help="Root containing sections/<section>/<section>.docling.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory; it must not already exist.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        manifest = read_book_manifest(arguments.manifest)
        if arguments.output.exists():
            raise ValueError(
                "output already exists; choose a new evidence version"
            )
        markers = _collect_formula_markers(
            manifest_sections=[
                section.section_id for section in manifest.sections
            ],
            conversion_root=arguments.conversion_root,
        )
        _validate_source(
            source=arguments.source,
            expected_sha256=manifest.source_sha256,
        )
        evidence = _extract_native_formula_text(
            source=arguments.source,
            markers=markers,
            output_directory=arguments.output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(
        "formula markers="
        f"{len(evidence['formula_restorations'])} "
            f"pages={len({item['page'] for item in evidence['formula_restorations']})}"
    )
    print(f"output: {arguments.output.resolve()}")
    return 0


def _collect_formula_markers(
    *,
    manifest_sections: list[str],
    conversion_root: Path,
) -> list[FormulaMarker]:
    markers: list[FormulaMarker] = []
    sections_root = conversion_root / "sections"

    for section_id in manifest_sections:
        section_root = sections_root / section_id
        document_path = section_root / f"{section_id}.docling.json"
        markdown_path = section_root / f"{section_id}.md"
        if not document_path.is_file() or not markdown_path.is_file():
            raise ValueError(
                f"missing Docling conversion for section: {section_id}"
            )

        document = json.loads(
            document_path.read_text(encoding="utf-8")
        )
        marker_lines = [
            line_number
            for line_number, line in enumerate(
                markdown_path.read_text(
                    encoding="utf-8"
                ).splitlines(),
                start=1,
            )
            if "formula-not-decoded" in line
        ]
        section_markers = _section_formula_markers(
            section_id=section_id,
            document=document,
            marker_lines=marker_lines,
            artifact_directory=section_root,
        )
        markers.extend(section_markers)

    if not markers:
        raise ValueError("no formula-not-decoded markers found")
    return markers


def _section_formula_markers(
    *,
    section_id: str,
    document: dict[str, Any],
    marker_lines: list[int],
    artifact_directory: Path,
) -> list[FormulaMarker]:
    texts = document.get("texts")
    body = document.get("body")
    if not isinstance(texts, list) or not isinstance(body, dict):
        raise ValueError(
            f"invalid Docling JSON for section: {section_id}"
        )

    body_refs = {
        child.get("$ref")
        for child in body.get("children", [])
        if isinstance(child, dict)
    }
    raw_markers = [
        (index, item)
        for index, item in enumerate(texts)
        if isinstance(item, dict)
        and item.get("label") == "formula"
        and not str(item.get("text") or "").strip()
        and item.get("self_ref") in body_refs
    ]
    if len(raw_markers) != len(marker_lines):
        raise ValueError(
            f"formula marker count does not match Markdown in {section_id}: "
            f"objects={len(raw_markers)} markers={len(marker_lines)}"
        )

    (
        raw_blocks,
        _,
        _,
        chapter_markers,
    ) = _extract_raw_blocks(
        document,
        artifact_directory=artifact_directory,
    )
    ordered_blocks, _ = _restore_single_column_order(raw_blocks)
    candidates, _, _ = _build_candidates(
        ordered_blocks,
        chapter_markers=chapter_markers,
    )
    candidate_positions = {
        block.source_ref: _raw_position_from_block(block)
        for block in ordered_blocks
    }

    results: list[FormulaMarker] = []
    for (index, item), marker_line in zip(
        raw_markers,
        marker_lines,
    ):
        marker_position = _raw_position_from_item(
            document=document,
            index=index,
            item=item,
        )
        anchor_index = max(
            (
                candidate_index
                for candidate_index, candidate in enumerate(candidates)
                if candidate_positions.get(candidate.source_ref)
                and candidate_positions[candidate.source_ref]
                < marker_position
            ),
            default=-1,
        )
        if anchor_index < 1 or anchor_index + 1 >= len(candidates):
            raise ValueError(
                f"could not anchor formula marker in {section_id}"
            )

        provenance = item.get("prov")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError(
                f"formula marker has no provenance in {section_id}"
            )
        page = int(provenance[0]["page_no"])
        bbox = provenance[0].get("bbox")
        if not isinstance(bbox, dict):
            raise ValueError(
                f"formula marker has no bounding box in {section_id}"
            )
        anchor = candidates[anchor_index]
        results.append(
            FormulaMarker(
                section_id=section_id,
                page=page,
                marker_line=marker_line,
                marker_source_ref=str(item["self_ref"]),
                anchor_source_ref=anchor.source_ref,
                before_text=candidates[anchor_index - 1].text,
                after_text=candidates[anchor_index + 1].text,
                bbox=(
                    float(bbox["l"]),
                    float(bbox["t"]),
                    float(bbox["r"]),
                    float(bbox["b"]),
                ),
            )
        )

    return results


def _raw_position_from_block(block: Any) -> tuple[int, float, float, int]:
    return (
        block.page_start,
        block.top_from_page,
        block.left,
        block.original_index,
    )


def _raw_position_from_item(
    *,
    document: dict[str, Any],
    index: int,
    item: dict[str, Any],
) -> tuple[int, float, float, int]:
    provenance = item.get("prov")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError(f"formula marker {index} has no provenance")
    first = provenance[0]
    if not isinstance(first, dict):
        raise ValueError(f"formula marker {index} has invalid provenance")
    page = int(first["page_no"])
    bbox = first.get("bbox")
    pages = document.get("pages")
    if not isinstance(bbox, dict) or not isinstance(pages, dict):
        raise ValueError(f"formula marker {index} has no bounding box")
    page_record = pages.get(str(page))
    if not isinstance(page_record, dict):
        raise ValueError(f"formula marker {index} references missing page")
    size = page_record.get("size")
    if not isinstance(size, dict):
        raise ValueError(f"formula marker {index} page has no size")
    top = float(bbox["t"])
    if bbox.get("coord_origin", "TOPLEFT") == "BOTTOMLEFT":
        top = float(size["height"]) - top
    return (page, top, float(bbox["l"]), index)


def _extract_native_formula_text(
    *,
    source: Path,
    markers: list[FormulaMarker],
    output_directory: Path,
) -> dict[str, object]:
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required; install rag-lab[conversion]"
        ) from error

    output_directory.mkdir(parents=True, exist_ok=False)
    crops_directory = output_directory / "formula-crops"
    crops_directory.mkdir()
    pdf = pymupdf.open(source)
    restorations: list[dict[str, object]] = []
    for marker in markers:
        page = pdf[marker.page - 1]
        replacement, native_blocks = _extract_formula_region(
            pymupdf=pymupdf,
            page=page,
            marker=marker,
        )
        crop_path = _write_formula_crop(
            pymupdf=pymupdf,
            page=page,
            marker=marker,
            output_directory=crops_directory,
        )
        restorations.append(
            {
                "section_id": marker.section_id,
                "page": marker.page,
                "marker_line": marker.marker_line,
                "marker_source_ref": marker.marker_source_ref,
                "anchor_source_ref": marker.anchor_source_ref,
                "before_text": marker.before_text,
                "after_text": marker.after_text,
                "replacement": replacement,
                "native_text_blocks": native_blocks,
                "formula_crop": crop_path.name,
            }
        )

    evidence: dict[str, object] = {
        "schema_version": "1.0",
        "extraction": {
            "engine": "pymupdf",
            "source": "original_pdf_native_text_layer",
        },
        "formula_restorations": restorations,
    }
    (output_directory / "formula-native-text.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def _write_formula_crop(
    *,
    pymupdf: Any,
    page: Any,
    marker: FormulaMarker,
    output_directory: Path,
) -> Path:
    page_height = page.rect.height
    left, top, right, bottom = marker.bbox
    formula_rect = pymupdf.Rect(
        left,
        page_height - top,
        right,
        page_height - bottom,
    )
    crop_rect = formula_rect + (-8, -8, 8, 8)
    crop_rect &= page.rect
    filename = (
        f"{marker.section_id}-p{marker.page:03d}-"
        f"{marker.marker_source_ref.rsplit('/', 1)[-1]}.png"
    )
    output_path = output_directory / filename
    page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        clip=crop_rect,
        alpha=False,
    ).save(output_path)
    return output_path


def _extract_formula_region(
    *,
    pymupdf: Any,
    page: Any,
    marker: FormulaMarker,
) -> tuple[str, list[str]]:
    page_height = page.rect.height
    left, top, right, bottom = marker.bbox
    formula_rect = pymupdf.Rect(
        left,
        page_height - top,
        right,
        page_height - bottom,
    )
    native_blocks = [
        str(block[4]).strip()
        for block in page.get_text("blocks")
        if pymupdf.Rect(block[:4]).intersects(formula_rect)
    ]
    formula_lines: list[str] = []
    for block in native_blocks:
        for line in block.splitlines():
            prefix, has_cjk = _formula_prefix(line)
            if prefix:
                formula_lines.append(prefix)
            if has_cjk:
                break
    if not formula_lines:
        raise ValueError(
            f"native PDF text has no formula content on page {marker.page}"
        )
    return "\n".join(formula_lines), native_blocks


def _formula_prefix(value: str) -> tuple[str, bool]:
    for index, char in enumerate(value):
        if "\u3400" <= char <= "\u9fff":
            if index == 0 and _looks_like_equation(value):
                return value.strip(), False
            return value[:index].strip(), True
    return value.strip(), False


def _looks_like_equation(value: str) -> bool:
    return any(
        character in value
        for character in ("=", "≥", "≤", "+", "−", "/", "{")
    )


def _validate_source(*, source: Path, expected_sha256: str) -> None:
    import hashlib

    digest_builder = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise ValueError("source SHA-256 does not match manifest")


if __name__ == "__main__":
    raise SystemExit(main())
