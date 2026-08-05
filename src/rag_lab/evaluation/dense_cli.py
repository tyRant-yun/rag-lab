from __future__ import annotations

import argparse
import json
import sys
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path

from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.evaluation import (
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    read_retrieval_evaluation_cases_jsonl,
)
from rag_lab.evaluation.evaluator import (
    RetrievalSearcher,
)
from rag_lab.retrieval.dense import (
    DenseRetriever,
)
from rag_lab.vector_store import (
    QdrantVectorStoreError,
    VectorStore,
)
from rag_lab.vector_store.cli import (
    build_qdrant_store,
)


ProviderFactory = Callable[
    ...,
    EmbeddingProvider,
]
StoreFactory = Callable[
    ...,
    VectorStore,
]
RetrieverFactory = Callable[
    ...,
    RetrievalSearcher,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-dense",
        description=(
            "Evaluate dense retrieval against "
            "a labeled JSONL query set."
        ),
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
        "--collection",
        required=True,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Evaluation cutoff. Default: 5.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help=(
            "Qdrant server URL. Default: "
            "http://localhost:6333."
        ),
    )
    parser.add_argument(
        "--model",
        default=(
            OllamaEmbeddingProvider.DEFAULT_MODEL
        ),
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--host",
        default=(
            OllamaEmbeddingProvider.DEFAULT_HOST
        ),
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=(
            OllamaEmbeddingProvider
            .DEFAULT_DIMENSIONS
        ),
        help="Embedding dimensions. Default: 1024.",
    )
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        default=60.0,
        help="Ollama request timeout.",
    )
    parser.add_argument(
        "--qdrant-timeout-seconds",
        type=int,
        default=10,
        help="Qdrant request timeout.",
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
    *,
    provider_factory: ProviderFactory = (
        OllamaEmbeddingProvider
    ),
    store_factory: StoreFactory = (
        build_qdrant_store
    ),
    retriever_factory: RetrieverFactory = (
        DenseRetriever
    ),
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        cases = (
            read_retrieval_evaluation_cases_jsonl(
                arguments.cases
            )
        )

        provider = provider_factory(
            model_name=arguments.model,
            dimensions=arguments.dimensions,
            host=arguments.host,
            timeout_seconds=(
                arguments.embedding_timeout_seconds
            ),
        )
        store = store_factory(
            url=arguments.url,
            collection_name=arguments.collection,
            dimensions=arguments.dimensions,
            timeout_seconds=(
                arguments.qdrant_timeout_seconds
            ),
        )
        retriever = retriever_factory(
            provider=provider,
            store=store,
        )
        report = RetrievalEvaluator().evaluate(
            dataset_id=arguments.dataset_id,
            cases=cases,
            retriever=retriever,
            top_k=arguments.top_k,
        )
    except (
        OSError,
        OllamaEmbeddingError,
        QdrantVectorStoreError,
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
