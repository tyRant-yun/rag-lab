from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path
from time import perf_counter

from rag_lab.contracts import (
    EmbeddingRunReport,
    KnowledgeChunk,
)
from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval import (
    read_knowledge_chunks_jsonl,
)


ProviderFactory = Callable[
    ...,
    EmbeddingProvider,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embed-chunks",
        description=(
            "Embed KnowledgeChunk index_text values "
            "and report validation statistics."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to KnowledgeChunk JSONL.",
    )
    parser.add_argument(
        "--model",
        default=(
            OllamaEmbeddingProvider.DEFAULT_MODEL
        ),
        help=(
            "Ollama embedding model. "
            "Default: qwen3-embedding:0.6b."
        ),
    )
    parser.add_argument(
        "--host",
        default=(
            OllamaEmbeddingProvider.DEFAULT_HOST
        ),
        help=(
            "Ollama server URL. "
            "Default: http://localhost:11434."
        ),
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
        "--batch-size",
        type=int,
        default=8,
        help="Chunks per embedding request. Default: 8.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Ollama request timeout. Default: 60.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write the complete report as JSON.",
    )

    return parser


def run_embedding_check(
    *,
    chunks: Sequence[KnowledgeChunk],
    provider: EmbeddingProvider,
    batch_size: int,
) -> EmbeddingRunReport:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
    ):
        raise TypeError(
            "batch_size must be an integer"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    if not chunks:
        raise ValueError(
            "chunks cannot be empty"
        )

    started_at = perf_counter()

    vector_norms: list[float] = []
    vector_count = 0
    batch_count = 0

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        current_chunks = chunks[
            start:start + batch_size
        ]
        texts = [
            chunk.index_text
            for chunk in current_chunks
        ]

        batch = provider.embed_documents(texts)

        if batch.provider != provider.provider_name:
            raise ValueError(
                "embedding batch provider mismatch"
            )

        if batch.model != provider.model_name:
            raise ValueError(
                "embedding batch model mismatch"
            )

        if batch.dimensions != provider.dimensions:
            raise ValueError(
                "embedding batch dimensions mismatch"
            )

        if (
            batch.embedding_version
            != provider.embedding_version
        ):
            raise ValueError(
                "embedding batch version mismatch"
            )

        if batch.input_count != len(current_chunks):
            raise ValueError(
                "embedding batch input count mismatch"
            )

        batch_count += 1
        vector_count += len(batch.vectors)

        for vector in batch.vectors:
            vector_norms.append(
                math.sqrt(
                    math.fsum(
                        value * value
                        for value in vector.values
                    )
                )
            )

    elapsed_ms = max(
        (
            perf_counter()
            - started_at
        )
        * 1000,
        0.0,
    )

    return EmbeddingRunReport(
        provider=provider.provider_name,
        model=provider.model_name,
        dimensions=provider.dimensions,
        embedding_version=(
            provider.embedding_version
        ),
        chunk_count=len(chunks),
        batch_count=batch_count,
        vector_count=vector_count,
        elapsed_ms=elapsed_ms,
        minimum_vector_norm=min(vector_norms),
        maximum_vector_norm=max(vector_norms),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: ProviderFactory = (
        OllamaEmbeddingProvider
    ),
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

        provider = provider_factory(
            model_name=arguments.model,
            dimensions=arguments.dimensions,
            host=arguments.host,
            timeout_seconds=(
                arguments.timeout_seconds
            ),
        )

        report = run_embedding_check(
            chunks=chunks,
            provider=provider,
            batch_size=arguments.batch_size,
        )
    except (
        OSError,
        OllamaEmbeddingError,
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
    report: EmbeddingRunReport,
) -> str:
    return "\n".join(
        [
            f"Provider: {report.provider}",
            f"Model: {report.model}",
            (
                "Embedding version: "
                f"{report.embedding_version}"
            ),
            f"Dimensions: {report.dimensions}",
            f"Chunks: {report.chunk_count}",
            f"Batches: {report.batch_count}",
            f"Vectors: {report.vector_count}",
            (
                "Vector norm range: "
                f"{report.minimum_vector_norm:.6f}"
                " - "
                f"{report.maximum_vector_norm:.6f}"
            ),
            (
                f"Elapsed: "
                f"{report.elapsed_ms:.2f} ms"
            ),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
