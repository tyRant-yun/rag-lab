from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockType(str, Enum):
    DOCUMENT_TITLE = "document_title"
    SECTION_HEADING = "section_heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    CODE = "code"
    EQUATION = "equation"


@dataclass(frozen=True, slots=True)
class NormalizedBlock:
    document_id: str
    text: str
    block_type: BlockType
    heading_path: tuple[str, ...]
    page_start: int
    page_end: int
    ordinal: int
    source_path: str
    normalization_version: str

    def __post_init__(self) -> None:
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

        if (
            self.block_type
            in {
                BlockType.DOCUMENT_TITLE,
                BlockType.SECTION_HEADING,
            }
            and self.heading_path[-1] != self.text
        ):
            raise ValueError(
                "heading blocks must end their "
                "heading_path with their text"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "block_type": self.block_type.value,
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "ordinal": self.ordinal,
            "source_path": self.source_path,
            "normalization_version": (
                self.normalization_version
            ),
        }


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

