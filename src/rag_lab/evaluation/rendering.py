from __future__ import annotations

from rag_lab.evaluation import (
    RetrievalEvaluationReport,
)


def render_human_report(
    report: RetrievalEvaluationReport,
) -> str:
    """Render one retrieval evaluation report for CLI output."""

    lines = [
        f"Dataset: {report.dataset_id}",
        f"Retriever: {report.retriever}",
        f"Index version: {report.index_version}",
        f"Top K: {report.top_k}",
        f"Cases: {report.case_count}",
        (
            f"Hit@{report.top_k}: "
            f"{report.hit_rate_at_k:.6f}"
        ),
        (
            f"Mean Recall@{report.top_k}: "
            f"{report.mean_recall_at_k:.6f}"
        ),
        f"MRR: {report.mrr:.6f}",
    ]

    for result in report.case_results:
        first_rank = (
            "-"
            if result.first_relevant_rank is None
            else str(result.first_relevant_rank)
        )
        hit = "yes" if result.hit_at_k else "no"

        lines.extend(
            [
                "",
                f"[{result.case_id}]",
                f"query={result.query}",
                f"hit={hit}",
                f"first_relevant_rank={first_rank}",
                (
                    f"recall@{result.top_k}="
                    f"{result.recall_at_k:.6f}"
                ),
                (
                    "reciprocal_rank="
                    f"{result.reciprocal_rank:.6f}"
                ),
                (
                    "relevant_chunk_ids="
                    + ", ".join(
                        result.relevant_chunk_ids
                    )
                ),
                (
                    "retrieved_chunk_ids="
                    + ", ".join(
                        result.retrieved_chunk_ids
                    )
                ),
            ]
        )

    return "\n".join(lines)
