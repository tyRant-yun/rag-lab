from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_lab.evaluation import (
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    read_retrieval_evaluation_cases_jsonl,
)
from rag_lab.retrieval import (
    read_knowledge_chunks_jsonl,
)
from rag_lab.retrieval.bm25 import (
    BM25Index,
    BM25Retriever,
)
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-bm25",
        description=(
            "Evaluate BM25 retrieval against "
            "a labeled JSONL query set."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to KnowledgeChunk JSONL.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help=(
            "Path to retrieval evaluation "
            "case JSONL."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable identifier for the dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Evaluation cutoff. Default: 5.",
    )
    parser.add_argument(
        "--user-word",
        action="append",
        dest="user_words",
        help=(
            "Add a domain-specific Jieba word. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--stopword",
        action="append",
        dest="stopwords",
        help=(
            "Exclude one lexical term. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write the complete report as JSON.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        chunks = read_knowledge_chunks_jsonl(
            arguments.chunks
        )

        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        cases = (
            read_retrieval_evaluation_cases_jsonl(
                arguments.cases,
                known_chunk_ids={
                    chunk.chunk_id
                    for chunk in chunks
                },
            )
        )

        analyzer = LexicalAnalyzer(
            user_words=arguments.user_words or (),
            stopwords=arguments.stopwords or (),
        )
        index = BM25Index(
            chunks=chunks,
            analyzer=analyzer,
        )
        retriever = BM25Retriever(index=index)

        report = RetrievalEvaluator().evaluate(
            dataset_id=arguments.dataset_id,
            cases=cases,
            retriever=retriever,
            top_k=arguments.top_k,
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        return 2

    if arguments.json_output:
        print(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_render_human_report(report))

    return 0


def _render_human_report(
    report: RetrievalEvaluationReport,
) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
