"""Public, product-level metadata for one searchable knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class PublicKnowledgeBaseInfo(BaseModel):
    """Safe metadata a browser may display about the available corpus."""

    model_config = ConfigDict(extra="forbid")

    title: str
    coverage: str
    topics: list[str]
    capabilities: list[str]
    guidance: list[str]
    limitations: list[str]


def read_public_knowledge_base_info(
    path: Path,
) -> PublicKnowledgeBaseInfo:
    """Read the public subset of a generated corpus manifest."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid knowledge-base manifest JSON: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "knowledge-base manifest must be a JSON object"
        )

    public_metadata = payload.get("public_metadata")
    if not isinstance(public_metadata, dict):
        raise ValueError(
            "knowledge-base manifest has no public_metadata object"
        )

    return PublicKnowledgeBaseInfo.model_validate(
        public_metadata,
        strict=True,
    )
