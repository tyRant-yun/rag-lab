from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
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
        prog="search-bm25",
        description=(
            "Search KnowledgeChunk JSONL files "
            "with an in-memory BM25 index."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to a KnowledgeChunk JSONL file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of hits. Default: 5.",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help=(
            "Restrict results to a document ID. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--heading",
        action="append",
        dest="heading_prefix",
        help=(
            "Add one heading-prefix component. "
            "Order is significant."
        ),
    )
    parser.add_argument(
        "--page-start",
        type=int,
        help="First source page to include.",
    )
    parser.add_argument(
        "--page-end",
        type=int,
        help="Last source page to include.",
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
        help="Write the complete SearchResult as JSON.",
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

        analyzer = LexicalAnalyzer(
            user_words=arguments.user_words or (),
            stopwords=arguments.stopwords or (),
        )
        index = BM25Index(
            chunks=chunks,
            analyzer=analyzer,
        )
        retriever = BM25Retriever(index=index)

        filters = SearchFilters(
            document_ids=arguments.document_ids,
            heading_prefix=arguments.heading_prefix,
            page_start=arguments.page_start,
            page_end=arguments.page_end,
        )

        result = retriever.search(
            arguments.query,
            top_k=arguments.top_k,
            filters=filters,
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
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_render_human_result(result))

    return 0


def _render_human_result(
    result: SearchResult,
) -> str:
    lines = [
        f"Query: {result.query}",
        f"Retriever: {result.retriever}",
        f"Index version: {result.index_version}",
        f"Candidates: {result.candidate_count}",
        f"Hits: {len(result.hits)}",
        f"Elapsed: {result.elapsed_ms:.2f} ms",
    ]

    if not result.hits:
        lines.extend(
            [
                "",
                "No matching chunks.",
            ]
        )
        return "\n".join(lines)

    for hit in result.hits:
        chunk = hit.chunk
        heading = " > ".join(
            chunk.heading_path
        )

        lines.extend(
            [
                "",
                (
                    f"[{hit.rank}] "
                    f"score={hit.score:.6f}"
                ),
                f"chunk_id={chunk.chunk_id}",
                (
                    f"pages={chunk.page_start}-"
                    f"{chunk.page_end}"
                ),
                f"heading={heading}",
                f"source={chunk.source_path}",
                "",
                chunk.content,
            ]
        )

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
