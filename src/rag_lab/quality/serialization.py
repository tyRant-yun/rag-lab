from __future__ import annotations

import json
from pathlib import Path

from rag_lab.quality.models import ArtifactQualityReport


def write_artifact_quality_report(
    *,
    report: ArtifactQualityReport,
    output_path: Path,
) -> None:
    """Write a stable, human-inspectable JSON report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
