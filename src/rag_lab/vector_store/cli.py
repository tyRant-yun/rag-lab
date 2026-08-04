from __future__ import annotations

import argparse
import json
import sys
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path

from qdrant_client import QdrantClient

from rag_lab.contracts import (
    KnowledgeChunk,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval import (
    read_knowledge_chunks_jsonl,
)
from rag_lab.vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreError,
    VectorStore,
)


ProviderFactory = Callable[
    ...,
    EmbeddingProvider,
]
StoreFactory = Callable[
    ...,
    VectorStore,
]


def build_qdrant_store(
    *,
    url: str,
    collection_name: str,
    dimensions: int,
    timeout_seconds: int,
) -> VectorStore:
    client = QdrantClient(
        url=url,
        timeout=timeout_seconds,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        dimensions=dimensions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="index-qdrant",
        description=(
            "Embed KnowledgeChunk index_text "
            "values and upsert them into Qdrant."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to KnowledgeChunk JSONL.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name.",
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
        "--batch-size",
        type=int,
        default=8,
        help="Chunks per embedding request.",
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
        help="Write the report as JSON.",
    )

    return parser


def index_chunks(
    *,
    chunks: Sequence[KnowledgeChunk],
    provider: EmbeddingProvider,
    store: VectorStore,
    batch_size: int,
) -> VectorWriteReport:
    if not chunks:
        raise ValueError(
            "chunks cannot be empty"
        )

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

    if provider.dimensions != store.dimensions:
        raise ValueError(
            "embedding provider dimensions must "
            "match vector store dimensions"
        )

    records: list[VectorRecord] = []

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        current_chunks = chunks[
            start:start + batch_size
        ]

        batch = provider.embed_documents(
            [
                chunk.index_text
                for chunk in current_chunks
            ]
        )

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

        records.extend(
            VectorRecord(
                chunk=chunk,
                vector=vector,
                embedding_version=(
                    batch.embedding_version
                ),
            )
            for chunk, vector in zip(
                current_chunks,
                batch.vectors,
                strict=True,
            )
        )

    return store.upsert(records)


def main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: ProviderFactory = (
        OllamaEmbeddingProvider
    ),
    store_factory: StoreFactory = (
        build_qdrant_store
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
                arguments
                .embedding_timeout_seconds
            ),
        )

        store = store_factory(
            url=arguments.url,
            collection_name=(
                arguments.collection
            ),
            dimensions=arguments.dimensions,
            timeout_seconds=(
                arguments
                .qdrant_timeout_seconds
            ),
        )

        report = index_chunks(
            chunks=chunks,
            provider=provider,
            store=store,
            batch_size=arguments.batch_size,
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
        print(
            _render_human_report(report)
        )

    return 0


def _render_human_report(
    report: VectorWriteReport,
) -> str:
    return "\n".join(
        [
            (
                "Collection: "
                f"{report.collection_name}"
            ),
            (
                "Dimensions: "
                f"{report.dimensions}"
            ),
            (
                "Input records: "
                f"{report.input_count}"
            ),
            (
                "Upserted records: "
                f"{report.upserted_count}"
            ),
            (
                "Embedding version: "
                f"{report.embedding_version}"
            ),
            (
                "Qdrant write elapsed: "
                f"{report.elapsed_ms:.2f} ms"
            ),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
