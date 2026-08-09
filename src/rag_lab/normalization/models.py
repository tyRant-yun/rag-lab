from __future__ import annotations

from dataclasses import dataclass

from rag_lab.contracts.blocks import NormalizedBlock


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
    downgraded_heading_count: int
    short_fragment_ratio: float
    pages_requiring_review: tuple[int, ...]
    correction_summary: dict[str, int] | None = None
    correction_overlay: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
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

        if self.correction_summary is not None:
            payload["correction_summary"] = dict(
                self.correction_summary
            )
        if self.correction_overlay is not None:
            payload["correction_overlay"] = dict(
                self.correction_overlay
            )

        return payload


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    blocks: tuple[NormalizedBlock, ...]
    report: NormalizationReport
