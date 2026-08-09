from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path

import uvicorn

from rag_lab.api.app import create_app
from rag_lab.embeddings import (
    OllamaEmbeddingProvider,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve-api",
        description=(
            "Run the RAG Lab retrieval API "
            "with uvicorn."
        ),
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=os.getenv("RAG_CHUNKS_PATH"),
        help="Path to KnowledgeChunk JSONL (or RAG_CHUNKS_PATH).",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("RAG_COLLECTION"),
        help="Qdrant collection name (or RAG_COLLECTION).",
    )
    parser.add_argument(
        "--bind-host",
        default=os.getenv("RAG_BIND_HOST", "127.0.0.1"),
        help="Uvicorn bind host (or RAG_BIND_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RAG_PORT", "8000")),
        help="Uvicorn port (or RAG_PORT).",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(
            "RAG_QDRANT_URL",
            "http://localhost:6333",
        ),
        help=(
            "Qdrant server URL. Default: "
            "http://localhost:6333."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "RAG_OLLAMA_MODEL",
            OllamaEmbeddingProvider.DEFAULT_MODEL,
        ),
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv(
            "RAG_OLLAMA_HOST",
            OllamaEmbeddingProvider.DEFAULT_HOST,
        ),
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=int(
            os.getenv(
                "RAG_EMBEDDING_DIMENSIONS",
                str(
                    OllamaEmbeddingProvider
                    .DEFAULT_DIMENSIONS
                ),
            )
        ),
        help="Embedding dimensions. Default: 1024.",
    )
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "RAG_EMBEDDING_TIMEOUT_SECONDS",
                "60",
            )
        ),
        help="Ollama request timeout.",
    )
    parser.add_argument(
        "--qdrant-timeout-seconds",
        type=int,
        default=int(
            os.getenv(
                "RAG_QDRANT_TIMEOUT_SECONDS",
                "10",
            )
        ),
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
        "--enable-debug-routes",
        action=argparse.BooleanOptionalAction,
        default=os.getenv(
            "RAG_ENABLE_DEBUG_ROUTES",
            "",
        ).lower()
        in {"1", "true", "yes"},
        help=(
            "Enable /search, /docs and /openapi.json. Disabled by default; "
            "set only for local development."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.chunks is None:
        raise SystemExit("--chunks or RAG_CHUNKS_PATH is required")
    if not arguments.collection:
        raise SystemExit("--collection or RAG_COLLECTION is required")
    app = create_app(
        chunks_path=arguments.chunks,
        collection=arguments.collection,
        url=arguments.url,
        model=arguments.model,
        host=arguments.ollama_host,
        dimensions=arguments.dimensions,
        embedding_timeout_seconds=(
            arguments
            .embedding_timeout_seconds
        ),
        qdrant_timeout_seconds=(
            arguments.qdrant_timeout_seconds
        ),
        user_words=arguments.user_words or (),
        stopwords=arguments.stopwords or (),
        enable_debug_routes=arguments.enable_debug_routes,
    )
    uvicorn.run(
        app,
        host=arguments.bind_host,
        port=arguments.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
