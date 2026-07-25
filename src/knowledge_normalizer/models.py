from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class BlockType(str, Enum):
    DOCUMENT_TITLE = "document_title"
    SECTION_HEADING = "section_heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    CODE = "code"
    EQUATION = "equation"


class NormalizedBlock(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    block_id: str
    document_id: str
    text: str
    block_type: str
    heading_path: list[str]
    page_start: int
    page_end: int
    ordinal: int
    source_path: str
    image_path: str | None
    normalization_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.block_id.strip():
            raise ValueError("block_id cannot be empty")

        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")

        if not self.text.strip():
            raise ValueError("text cannot be empty")

        if not self.heading_path:
            raise ValueError("heading_path cannot be empty")

        if any(
            not heading.strip()
            for heading in self.heading_path
        ):
            raise ValueError(
                "heading_path entries cannot be empty"
            )

        if self.page_start < 1:
            raise ValueError(
                "page_start must be at least 1"
            )

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end must not precede page_start"
            )

        if self.ordinal < 1:
            raise ValueError(
                "ordinal must be at least 1"
            )

        if not self.source_path.strip():
            raise ValueError("source_path cannot be empty")

        if not self.normalization_version.strip():
            raise ValueError(
                "normalization_version cannot be empty"
            )

        valid_block_types = {
            member.value
            for member in BlockType
        }

        if self.block_type not in valid_block_types:
            raise ValueError(
                f"unsupported block_type: "
                f"{self.block_type}"
            )

        if (
            self.block_type
            in {
                BlockType.DOCUMENT_TITLE.value,
                BlockType.SECTION_HEADING.value,
            }
            and self.heading_path[-1] != self.text
        ):
            raise ValueError(
                "heading blocks must end their "
                "heading_path with their text"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class NormalizationReport:
    document_id: str
    normalization_version: str
    source_path: str
    source_pages: tuple[int, ...]
    raw_block_count: int
    normalized_block_count: int
    removed_furniture_count: int
    reordered_block_count: int
    merged_cross_page_count: int
    downgraded_heading_count: int
    short_fragment_ratio: float
    pages_requiring_review: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "normalization_version": (
                self.normalization_version
            ),
            "source_path": self.source_path,
            "source_pages": list(self.source_pages),
            "raw_block_count": self.raw_block_count,
            "normalized_block_count": (
                self.normalized_block_count
            ),
            "removed_furniture_count": (
                self.removed_furniture_count
            ),
            "reordered_block_count": (
                self.reordered_block_count
            ),
            "merged_cross_page_count": (
                self.merged_cross_page_count
            ),
            "downgraded_heading_count": (
                self.downgraded_heading_count
            ),
            "short_fragment_ratio": round(
                self.short_fragment_ratio,
                6,
            ),
            "pages_requiring_review": list(
                self.pages_requiring_review
            ),
        }


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    blocks: tuple[NormalizedBlock, ...]
    report: NormalizationReport
