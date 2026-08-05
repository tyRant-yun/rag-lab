from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rag_lab.embeddings import (
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval import (
    read_knowledge_chunks_jsonl,
)
from rag_lab.retrieval.bm25 import (
    BM25Index,
    BM25Retriever,
)
from rag_lab.retrieval.dense import (
    DenseRetriever,
)
from rag_lab.retrieval.hybrid import (
    HybridRetriever,
)
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)
from rag_lab.retrieval.rerank import (
    LexicalOverlapReranker,
    RerankedRetriever,
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
        chunks = read_knowledge_chunks_jsonl(
            Path(chunks_path)
        )

        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        analyzer = LexicalAnalyzer(
            user_words=user_words,
            stopwords=stopwords,
        )
        index = BM25Index(
            chunks=chunks,
            analyzer=analyzer,
        )
        bm25 = BM25Retriever(index=index)

        provider = provider_factory(
            model_name=model,
            dimensions=dimensions,
            host=host,
            timeout_seconds=(
                embedding_timeout_seconds
            ),
        )
        store = store_factory(
            url=url,
            collection_name=collection,
            dimensions=dimensions,
            timeout_seconds=qdrant_timeout_seconds,
        )
        dense = DenseRetriever(
            provider=provider,
            store=store,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            dense=dense,
        )
        reranker = LexicalOverlapReranker(
            analyzer=analyzer,
        )
        rerank = RerankedRetriever(
            retriever=hybrid,
            reranker=reranker,
        )

        tool = SearchKnowledgeTool(
            retrievers={
                "bm25": bm25,
                "dense": dense,
                "hybrid": hybrid,
                "rerank": rerank,
            }
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
