from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rag_lab.embeddings import (
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval.factory import (
    build_retrieval_components,
)
from rag_lab.tools.search_tool import (
    SearchKnowledgeTool,
)
from rag_lab.vector_store.cli import (
    build_qdrant_store,
)


class RetrievalToolset:
    """Assemble knowledge-base retrieval tools for an agent."""

    def __init__(
        self,
        *,
        tool: SearchKnowledgeTool,
    ) -> None:
        self._tool = tool

    @classmethod
    def build(
        cls,
        *,
        chunks_path: Path | str,
        collection: str,
        url: str = "http://localhost:6333",
        model: str = (
            OllamaEmbeddingProvider.DEFAULT_MODEL
        ),
        host: str = (
            OllamaEmbeddingProvider.DEFAULT_HOST
        ),
        dimensions: int = (
            OllamaEmbeddingProvider
            .DEFAULT_DIMENSIONS
        ),
        embedding_timeout_seconds: float = 60.0,
        qdrant_timeout_seconds: int = 10,
        user_words: Sequence[str] = (),
        stopwords: Sequence[str] = (),
        provider_factory=OllamaEmbeddingProvider,
        store_factory=build_qdrant_store,
    ) -> RetrievalToolset:
        components = build_retrieval_components(
            chunks_path=chunks_path,
            collection=collection,
            url=url,
            model=model,
            host=host,
            dimensions=dimensions,
            embedding_timeout_seconds=(
                embedding_timeout_seconds
            ),
            qdrant_timeout_seconds=(
                qdrant_timeout_seconds
            ),
            user_words=user_words,
            stopwords=stopwords,
            provider_factory=provider_factory,
            store_factory=store_factory,
        )

        tool = SearchKnowledgeTool(
            retrievers=components.all()
        )

        return cls(tool=tool)

    @property
    def tool(self) -> SearchKnowledgeTool:
        return self._tool

    def to_openai_tools(
        self,
    ) -> list[dict[str, object]]:
        return [
            self._tool.openai_schema()
        ]

    def execute(
        self,
        arguments: Mapping[str, object],
    ) -> dict[str, object]:
        return self._tool.execute_raw(arguments)
