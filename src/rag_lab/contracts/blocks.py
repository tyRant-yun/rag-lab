from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
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

        if self.image_path:
            portable_image_path = (
                PurePosixPath(self.image_path)
            )

            if (
                portable_image_path.is_absolute()
                or PureWindowsPath(
                    self.image_path
                ).is_absolute()
                or ".."
                in portable_image_path.parts
                or "\\" in self.image_path
            ):
                raise ValueError(
                    "image_path must be a portable "
                    "relative path"
                )

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
