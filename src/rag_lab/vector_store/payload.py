from __future__ import annotations

import json
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any
from uuid import (
    NAMESPACE_URL,
    uuid5,
)

from pydantic import ValidationError
from qdrant_client import models

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    VectorRecord,
)


_CHUNK_PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "content",
    "index_text",
    "heading_path",
    "page_start",
    "page_end",
    "ordinal",
    "block_ids",
    "source_path",
    "content_hash",
    "normalization_version",
    "chunking_version",
)


def chunk_point_id(chunk_id: str) -> str:
    """Return a stable Qdrant UUID for one Chunk ID."""

    if not isinstance(chunk_id, str):
        raise TypeError(
            "chunk_id must be a string"
        )

    if not chunk_id.strip():
        raise ValueError(
            "chunk_id cannot be empty"
        )

    return str(
        uuid5(
            NAMESPACE_URL,
            f"rag-lab:chunk:{chunk_id}",
        )
    )


def heading_prefix_key(
    heading_path: Sequence[str],
) -> str:
    """Encode one ordered heading path for filtering."""

    if isinstance(
        heading_path,
        (str, bytes),
    ):
        raise TypeError(
            "heading_path must be a sequence "
            "of strings"
        )

    active_path = list(heading_path)

    if not active_path:
        raise ValueError(
            "heading_path cannot be empty"
        )

    if any(
        not isinstance(heading, str)
        or not heading.strip()
        for heading in active_path
    ):
        raise ValueError(
            "heading_path entries must be "
            "non-empty strings"
        )

    return json.dumps(
        active_path,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def heading_prefix_keys(
    heading_path: Sequence[str],
) -> list[str]:
    """Return every ordered prefix of a heading path."""

    active_path = list(heading_path)

    if not active_path:
        raise ValueError(
            "heading_path cannot be empty"
        )

    return [
        heading_prefix_key(
            active_path[:prefix_length]
        )
        for prefix_length in range(
            1,
            len(active_path) + 1,
        )
    ]


def record_to_payload(
    record: VectorRecord,
) -> dict[str, Any]:
    """Serialize one VectorRecord as Qdrant payload."""

    if not isinstance(record, VectorRecord):
        raise TypeError(
            "record must be VectorRecord"
        )

    payload: dict[str, Any] = dict(
        record.chunk.to_dict()
    )

    payload["heading_prefixes"] = (
        heading_prefix_keys(
            record.chunk.heading_path
        )
    )
    payload["embedding_version"] = (
        record.embedding_version
    )

    return payload


def payload_to_chunk(
    payload: Mapping[str, Any],
) -> KnowledgeChunk:
    """Restore a validated KnowledgeChunk from payload."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "payload must be a mapping"
        )

    missing_fields = [
        field_name
        for field_name in _CHUNK_PAYLOAD_FIELDS
        if field_name not in payload
    ]

    if missing_fields:
        missing_text = ", ".join(
            sorted(missing_fields)
        )
        raise ValueError(
            "payload is missing KnowledgeChunk "
            f"fields: {missing_text}"
        )

    chunk_payload = {
        field_name: payload[field_name]
        for field_name
        in _CHUNK_PAYLOAD_FIELDS
    }

    try:
        return KnowledgeChunk.model_validate(
            chunk_payload,
            strict=True,
        )
    except ValidationError as error:
        raise ValueError(
            "payload contains an invalid "
            "KnowledgeChunk"
        ) from error


def filters_to_qdrant_filter(
    filters: SearchFilters | None,
) -> models.Filter | None:
    """Convert storage-neutral filters to Qdrant."""

    if filters is None:
        return None

    if not isinstance(filters, SearchFilters):
        raise TypeError(
            "filters must be SearchFilters"
        )

    must: list[models.FieldCondition] = []

    if filters.document_ids is not None:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(
                    any=filters.document_ids,
                ),
            )
        )

    if filters.heading_prefix is not None:
        must.append(
            models.FieldCondition(
                key="heading_prefixes",
                match=models.MatchValue(
                    value=heading_prefix_key(
                        filters.heading_prefix
                    ),
                ),
            )
        )

    if filters.page_start is not None:
        must.append(
            models.FieldCondition(
                key="page_end",
                range=models.Range(
                    gte=float(
                        filters.page_start
                    ),
                ),
            )
        )

    if filters.page_end is not None:
        must.append(
            models.FieldCondition(
                key="page_start",
                range=models.Range(
                    lte=float(
                        filters.page_end
                    ),
                ),
            )
        )

    if not must:
        return None

    return models.Filter(
        must=must,
    )
