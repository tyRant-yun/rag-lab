from __future__ import annotations

import argparse
import json
import sys
from collections.abc import (
    Callable,
    Sequence,
)

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval.dense.retriever import (
    DenseRetriever,
)
from rag_lab.retrieval.rendering import (
    render_human_result,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-dense",
        description=(
            "Search an existing Qdrant collection "
            "with dense vector retrieval."
        ),
    )

    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name.",
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
        "--json",
        action="store_true",
        dest="json_output",
        help="Write the complete SearchResult as JSON.",
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
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
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
        retriever = DenseRetriever(
            provider=provider,
            store=store,
        )
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
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_human_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
