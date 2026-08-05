from __future__ import annotations

import argparse
import json
import sys
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval.factory import (
    build_retrieval_components,
)
from rag_lab.retrieval.hybrid import (
    HybridRetriever,
)
from rag_lab.retrieval.rerank import (
    RerankedRetriever,
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


def _add_rerank_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=RerankedRetriever.DEFAULT_FETCH_K,
        help=(
            "Candidates fetched from the base retriever "
            "before reranking. Default: 20."
        ),
    )
    parser.add_argument(
        "--rrf-weight",
        type=float,
        default=1.0,
        help="Weight of the fused RRF score. Default: 1.0.",
    )
    parser.add_argument(
        "--overlap-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of query-term overlap with index_text. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--heading-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of query-term overlap with heading_path. "
            "Default: 1.0."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-rerank",
        description=(
            "Search with RRF hybrid retrieval, then rerank "
            "candidates by lexical overlap."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to KnowledgeChunk JSONL.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text.",
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
        help="Maximum number of hits. Default: 5.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=HybridRetriever.DEFAULT_RRF_K,
        help="RRF smoothing constant. Default: 60.",
    )
    parser.add_argument(
        "--per-retriever-k",
        type=int,
        default=(
            HybridRetriever.DEFAULT_PER_RETRIEVER_K
        ),
        help=(
            "Hits requested from each retriever "
            "before fusion. Default: 10."
        ),
    )
    _add_rerank_arguments(parser)
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
        components = build_retrieval_components(
            chunks_path=arguments.chunks,
            collection=arguments.collection,
            url=arguments.url,
            model=arguments.model,
            host=arguments.host,
            dimensions=arguments.dimensions,
            embedding_timeout_seconds=(
                arguments
                .embedding_timeout_seconds
            ),
            qdrant_timeout_seconds=(
                arguments
                .qdrant_timeout_seconds
            ),
            user_words=arguments.user_words or (),
            stopwords=arguments.stopwords or (),
            provider_factory=provider_factory,
            store_factory=store_factory,
        )
        retriever = components.retriever(
            "rerank",
            rrf_k=arguments.rrf_k,
            per_retriever_k=(
                arguments.per_retriever_k
            ),
            fetch_k=arguments.fetch_k,
            rrf_weight=arguments.rrf_weight,
            overlap_weight=arguments.overlap_weight,
            heading_weight=arguments.heading_weight,
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
