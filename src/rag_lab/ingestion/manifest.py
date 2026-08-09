"""Strict, versioned manifest models for one source book."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rag_lab.knowledge_base import (
    PublicKnowledgeBaseInfo,
)


class BookSection(BaseModel):
    """One non-overlapping physical-page range from a source PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    include_in_index: bool = True

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.end_page < self.start_page:
            raise ValueError(
                "end_page must not precede start_page"
            )
        return self


class BookManifest(BaseModel):
    """The declarative source and scope for a complete book build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    book_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    source_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    source_page_count: int = Field(ge=1)
    sections: list[BookSection] = Field(min_length=1)
    public_metadata: PublicKnowledgeBaseInfo

    @model_validator(mode="after")
    def validate_sections(self) -> Self:
        seen_ids: set[str] = set()
        last_end = 0

        for section in self.sections:
            if section.section_id in seen_ids:
                raise ValueError(
                    "section_id values must be unique"
                )
            seen_ids.add(section.section_id)

            if section.start_page <= last_end:
                raise ValueError(
                    "sections must be ordered and non-overlapping"
                )
            if section.end_page > self.source_page_count:
                raise ValueError(
                    "section page range exceeds source_page_count"
                )
            last_end = section.end_page

        return self


def read_book_manifest(path: Path) -> BookManifest:
    """Load one manifest without accepting unknown or malformed fields."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid book manifest JSON: {path}"
        ) from error

    try:
        return BookManifest.model_validate(
            payload,
            strict=True,
        )
    except ValidationError as error:
        raise ValueError(
            f"invalid book manifest: {path}"
        ) from error
