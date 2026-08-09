from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CorrectionOperation = Literal[
    "merge_text",
    "replace_text",
    "reorder_blocks",
    "reclassify_block",
    "exclude_from_index",
    "insert_equation",
]

_OPERATIONS = frozenset(
    {
        "merge_text",
        "replace_text",
        "reorder_blocks",
        "reclassify_block",
        "exclude_from_index",
        "insert_equation",
    }
)


@dataclass(frozen=True, slots=True)
class Correction:
    """One narrow, source-anchored normalization correction."""

    correction_id: str
    page: int
    operation: CorrectionOperation
    source_refs: tuple[str, ...]
    before_text: str
    after_text: str
    reason: str
    replacement: str | None = None
    find_text: str | None = None
    block_type: str | None = None
    marker_line: int | None = None
    marker_source_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CorrectionOverlay:
    """A document-specific, versioned list of corrections."""

    document_id: str
    schema_version: str
    sha256: str
    source_path: str
    corrections: tuple[Correction, ...]


def _required_string(
    value: object,
    *,
    field: str,
    context: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{context}.{field} must be a non-empty string"
        )

    return value


def _source_refs(
    value: object,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{context}.source_refs must be a non-empty array"
        )

    refs = tuple(
        _required_string(
            item,
            field="source_refs entry",
            context=context,
        )
        for item in value
    )
    if len(set(refs)) != len(refs):
        raise ValueError(
            f"{context}.source_refs cannot contain duplicates"
        )

    return refs


def _parse_correction(
    raw: object,
    *,
    index: int,
) -> Correction:
    context = f"corrections[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")

    operation = _required_string(
        raw.get("operation"),
        field="operation",
        context=context,
    )
    if operation not in _OPERATIONS:
        raise ValueError(
            f"{context}.operation is unsupported: {operation}"
        )

    page = raw.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError(
            f"{context}.page must be a positive integer"
        )

    replacement = raw.get("replacement")
    if replacement is not None and (
        not isinstance(replacement, str)
        or not replacement.strip()
    ):
        raise ValueError(
            f"{context}.replacement must be a non-empty string"
        )

    find_text = raw.get("find_text")
    if find_text is not None and (
        not isinstance(find_text, str)
        or not find_text
    ):
        raise ValueError(
            f"{context}.find_text must be a non-empty string"
        )

    block_type = raw.get("block_type")
    if block_type is not None and (
        not isinstance(block_type, str)
        or not block_type.strip()
    ):
        raise ValueError(
            f"{context}.block_type must be a non-empty string"
        )

    marker_line = raw.get("marker_line")
    if marker_line is not None and (
        isinstance(marker_line, bool)
        or not isinstance(marker_line, int)
        or marker_line < 1
    ):
        raise ValueError(
            f"{context}.marker_line must be a positive integer"
        )

    marker_source_ref = raw.get("marker_source_ref")
    if marker_source_ref is not None and (
        not isinstance(marker_source_ref, str)
        or not marker_source_ref.strip()
    ):
        raise ValueError(
            f"{context}.marker_source_ref must be a non-empty string"
        )

    if operation in {
        "merge_text",
        "replace_text",
        "insert_equation",
    } and replacement is None:
        raise ValueError(
            f"{context}.replacement is required for {operation}"
        )

    if (
        operation == "reclassify_block"
        and block_type is None
    ):
        raise ValueError(
            f"{context}.block_type is required for reclassify_block"
        )

    if operation == "insert_equation" and marker_line is None:
        raise ValueError(
            f"{context}.marker_line is required for insert_equation"
        )

    if operation == "insert_equation" and marker_source_ref is None:
        raise ValueError(
            f"{context}.marker_source_ref is required for insert_equation"
        )

    return Correction(
        correction_id=_required_string(
            raw.get("id"),
            field="id",
            context=context,
        ),
        page=page,
        operation=operation,  # type: ignore[arg-type]
        source_refs=_source_refs(
            raw.get("source_refs"),
            context=context,
        ),
        before_text=_required_string(
            raw.get("before_text"),
            field="before_text",
            context=context,
        ),
        after_text=_required_string(
            raw.get("after_text"),
            field="after_text",
            context=context,
        ),
        reason=_required_string(
            raw.get("reason"),
            field="reason",
            context=context,
        ),
        replacement=replacement,
        find_text=find_text,
        block_type=block_type,
        marker_line=marker_line,
        marker_source_ref=marker_source_ref,
    )


def read_correction_overlay(
    path: Path,
) -> CorrectionOverlay:
    """Load and validate one declarative correction overlay."""

    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid correction overlay JSON: {path}"
        ) from error

    if not isinstance(raw, dict):
        raise ValueError("correction overlay must be a JSON object")

    corrections = raw.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise ValueError(
            "correction overlay must contain corrections"
        )

    parsed = tuple(
        _parse_correction(item, index=index)
        for index, item in enumerate(corrections, start=1)
    )
    correction_ids = [
        correction.correction_id
        for correction in parsed
    ]
    if len(set(correction_ids)) != len(correction_ids):
        raise ValueError(
            "correction overlay correction IDs must be unique"
        )

    return CorrectionOverlay(
        document_id=_required_string(
            raw.get("document_id"),
            field="document_id",
            context="overlay",
        ),
        schema_version=_required_string(
            raw.get("schema_version"),
            field="schema_version",
            context="overlay",
        ),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_path=str(path.resolve()),
        corrections=parsed,
    )
