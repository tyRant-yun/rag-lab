"""Build section-local, source-anchored formula overlays from evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_lab.ingestion import read_book_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine reviewed formula replacements with source-bound evidence "
            "into per-section correction overlays."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replacements", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory for per-section overlay JSON files.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        manifest = read_book_manifest(arguments.manifest)
        replacements = _read_json(arguments.replacements)
        evidence = _read_json(arguments.evidence)
        overlays = _build_overlays(
            source_sha256=manifest.source_sha256,
            replacements=replacements,
            evidence=evidence,
        )
        _write_overlays(overlays=overlays, output=arguments.output)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2

    correction_count = sum(
        len(overlay["corrections"])
        for overlay in overlays.values()
    )
    print(
        f"sections={len(overlays)} formula_corrections={correction_count}"
    )
    print(f"output: {arguments.output.resolve()}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _build_overlays(
    *,
    source_sha256: str,
    replacements: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, dict[str, object]]:
    if replacements.get("source_sha256") != source_sha256:
        raise ValueError("replacement source SHA-256 does not match manifest")
    replacement_rows = replacements.get("replacements")
    evidence_rows = evidence.get("formula_restorations")
    if not isinstance(replacement_rows, list) or not isinstance(
        evidence_rows, list
    ):
        raise ValueError("replacement and evidence rows must be arrays")

    replacement_by_key = {
        _formula_key(row): row for row in replacement_rows
    }
    evidence_by_key = {
        _formula_key(row): row for row in evidence_rows
    }
    if len(replacement_by_key) != len(replacement_rows):
        raise ValueError("replacement rows contain duplicate formula markers")
    if len(evidence_by_key) != len(evidence_rows):
        raise ValueError("evidence rows contain duplicate formula markers")
    if replacement_by_key.keys() != evidence_by_key.keys():
        raise ValueError("replacement rows do not exactly match evidence markers")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(replacement_by_key):
        replacement_row = replacement_by_key[key]
        evidence_row = evidence_by_key[key]
        section_id, page, marker_source_ref = key
        replacement = _required_string(
            replacement_row.get("replacement"),
            field="replacement",
        )
        grouped[section_id].append(
            {
                "id": (
                    f"full-book-v1-formula-p{page}-"
                    f"{marker_source_ref.rsplit('/', 1)[-1]}"
                ),
                "page": page,
                "operation": "insert_equation",
                "marker_line": _required_integer(
                    evidence_row.get("marker_line"),
                    field="marker_line",
                ),
                "marker_source_ref": marker_source_ref,
                "source_refs": [
                    _required_string(
                        evidence_row.get("anchor_source_ref"),
                        field="anchor_source_ref",
                    )
                ],
                "before_text": _required_string(
                    evidence_row.get("before_text"),
                    field="before_text",
                ),
                "after_text": _required_string(
                    evidence_row.get("after_text"),
                    field="after_text",
                ),
                "replacement": replacement,
                "reason": (
                    "Restore this exact formula from the original PDF; "
                    "evidence binds the Markdown marker, page, marker "
                    "source ref, and insertion anchor."
                ),
            }
        )

    return {
        section_id: {
            "schema_version": "2.0",
            "document_id": f"sha256:{source_sha256}",
            "corrections": corrections,
        }
        for section_id, corrections in grouped.items()
    }


def _formula_key(row: object) -> tuple[str, int, str]:
    if not isinstance(row, dict):
        raise ValueError("formula row must be an object")
    return (
        _required_string(row.get("section_id"), field="section_id"),
        _required_integer(row.get("page"), field="page"),
        _required_string(
            row.get("marker_source_ref"),
            field="marker_source_ref",
        ),
    )


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _write_overlays(
    *,
    overlays: dict[str, dict[str, object]],
    output: Path,
) -> None:
    if output.exists():
        raise ValueError("output already exists; choose a new overlay version")
    output.mkdir(parents=True)
    for section_id, overlay in overlays.items():
        (output / f"{section_id}.json").write_text(
            json.dumps(overlay, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
