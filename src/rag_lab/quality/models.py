from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ArtifactQualityInputs:
    """The immutable input bundle consumed by one artifact audit."""

    docling_markdown: Path
    normalization_report: Path
    blocks: Path
    chunking_report: Path
    chunks: Path


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Conservative, explicit thresholds for deterministic checks."""

    max_chunk_chars: int = 1200
    short_fragment_ratio_warning: float = 0.04
    review_page_ratio_warning: float = 0.25

    def __post_init__(self) -> None:
        if self.max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be at least 1")

        for name, value in (
            (
                "short_fragment_ratio_warning",
                self.short_fragment_ratio_warning,
            ),
            (
                "review_page_ratio_warning",
                self.review_page_ratio_warning,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ArtifactQualityIssue:
    """One source-grounded issue emitted by the quality gate."""

    issue_code: str
    severity: IssueSeverity
    pipeline_stage: str
    document_id: str | None
    page: int | None
    block_id: str | None
    block_ordinal: int | None
    chunk_id: str | None
    chunk_ordinal: int | None
    evidence: str
    remediation_hint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "issue_code": self.issue_code,
            "severity": self.severity,
            "pipeline_stage": self.pipeline_stage,
            "document_id": self.document_id,
            "page": self.page,
            "block_id": self.block_id,
            "block_ordinal": self.block_ordinal,
            "chunk_id": self.chunk_id,
            "chunk_ordinal": self.chunk_ordinal,
            "evidence": self.evidence,
            "remediation_hint": self.remediation_hint,
        }


@dataclass(frozen=True, slots=True)
class ArtifactQualityReport:
    """A deterministic report generated from one read-only artifact bundle."""

    document_id: str | None
    inputs: ArtifactQualityInputs
    config: AuditConfig
    issues: tuple[ArtifactQualityIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity == "error"
            for issue in self.issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity == "warning"
            for issue in self.issues
        )

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "inputs": {
                "docling_markdown": str(
                    self.inputs.docling_markdown
                ),
                "normalization_report": str(
                    self.inputs.normalization_report
                ),
                "blocks": str(self.inputs.blocks),
                "chunking_report": str(
                    self.inputs.chunking_report
                ),
                "chunks": str(self.inputs.chunks),
            },
            "config": {
                "max_chunk_chars": self.config.max_chunk_chars,
                "short_fragment_ratio_warning": (
                    self.config.short_fragment_ratio_warning
                ),
                "review_page_ratio_warning": (
                    self.config.review_page_ratio_warning
                ),
            },
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }
