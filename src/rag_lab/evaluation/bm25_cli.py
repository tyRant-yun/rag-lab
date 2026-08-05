from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_lab.evaluation import (
    RetrievalEvaluator,
    read_retrieval_evaluation_cases_jsonl,
)
from rag_lab.evaluation.rendering import (
    render_human_report,
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
        print(render_human_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
