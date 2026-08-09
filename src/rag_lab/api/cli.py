from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from rag_lab.api.app import create_app
from rag_lab.embeddings import (
    OllamaEmbeddingProvider,
)
from rag_lab.knowledge_base import (
    read_public_knowledge_base_info,
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
        required=True,
        help="Path to KnowledgeChunk JSONL.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help="Uvicorn bind host. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Uvicorn port. Default: 8000.",
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
        "--ollama-host",
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
        "--knowledge-base-manifest",
        type=Path,
        help=(
            "Generated corpus-manifest.json whose public_metadata "
            "will be shown in the browser UI."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        knowledge_base_info = (
            read_public_knowledge_base_info(
                arguments.knowledge_base_manifest
            )
            if arguments.knowledge_base_manifest is not None
            else None
        )
    except (OSError, ValueError) as error:
        build_parser().error(str(error))
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
        enable_debug_routes=(
            arguments.bind_host in {"127.0.0.1", "localhost", "::1"}
        ),
        knowledge_base_info=knowledge_base_info,
    )
    uvicorn.run(
        app,
        host=arguments.bind_host,
        port=arguments.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
