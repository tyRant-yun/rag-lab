from __future__ import annotations

import argparse
import json
import sys
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from pathlib import Path

from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.tools.retrieval_toolset import (
    RetrievalToolset,
)
from rag_lab.tools.search_tool import (
    SearchKnowledgeTool,
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


def _add_toolset_arguments(
    parser: argparse.ArgumentParser,
) -> None:
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


def schema_main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="search-tool-schema",
        description=(
            "Print the OpenAI function schema of the "
            "search_knowledge tool."
        ),
    )
    parser.parse_args(argv)

    print(
        json.dumps(
            SearchKnowledgeTool.openai_schema(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def execute_main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: ProviderFactory = (
        OllamaEmbeddingProvider
    ),
    store_factory: StoreFactory = (
        build_qdrant_store
    ),
) -> int:
    parser = argparse.ArgumentParser(
        prog="execute-search-tool",
        description=(
            "Validate and execute the search_knowledge "
            "tool with JSON arguments."
        ),
    )
    _add_toolset_arguments(parser)
    parser.add_argument(
        "--args",
        required=True,
        help="Tool arguments as a JSON object string.",
    )
    arguments = parser.parse_args(argv)

    try:
        raw_arguments: object = json.loads(
            arguments.args
        )

        if not isinstance(
            raw_arguments,
            dict,
        ):
            raise TypeError(
                "--args must be a JSON object"
            )

        toolset = RetrievalToolset.build(
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
                arguments
                .qdrant_timeout_seconds
            ),
            user_words=arguments.user_words or (),
            stopwords=arguments.stopwords or (),
            provider_factory=provider_factory,
            store_factory=store_factory,
        )
        result = toolset.execute(
            dict(raw_arguments)
        )
    except (
        OSError,
        OllamaEmbeddingError,
        QdrantVectorStoreError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(execute_main())
