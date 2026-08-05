from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from rag_lab.vector_store.cli import (
    build_qdrant_store,
)


RetrieverName = Literal[
    "bm25",
    "dense",
    "hybrid",
    "rerank",
]
RETRIEVER_NAMES: tuple[
    RetrieverName,
    ...,
] = (
    "bm25",
    "dense",
    "hybrid",
    "rerank",
)


@dataclass(frozen=True)
class RetrievalComponents:
    """Shared retrieval infrastructure for CLIs, API and tools."""

    analyzer: LexicalAnalyzer
    bm25: BM25Retriever
    dense: DenseRetriever

    def retriever(
        self,
        name: RetrieverName,
        *,
        rrf_k: int = (
            HybridRetriever.DEFAULT_RRF_K
        ),
        per_retriever_k: int = (
            HybridRetriever
            .DEFAULT_PER_RETRIEVER_K
        ),
        fetch_k: int = (
            RerankedRetriever.DEFAULT_FETCH_K
        ),
        rrf_weight: float = 1.0,
        overlap_weight: float = 1.0,
        heading_weight: float = 1.0,
    ):
        if name == "bm25":
            return self.bm25

        if name == "dense":
            return self.dense

        hybrid = HybridRetriever(
            bm25=self.bm25,
            dense=self.dense,
            rrf_k=rrf_k,
            per_retriever_k=per_retriever_k,
        )

        if name == "hybrid":
            return hybrid

        reranker = LexicalOverlapReranker(
            analyzer=self.analyzer,
            rrf_weight=rrf_weight,
            overlap_weight=overlap_weight,
            heading_weight=heading_weight,
        )

        return RerankedRetriever(
            retriever=hybrid,
            reranker=reranker,
            fetch_k=fetch_k,
        )

    def all(self) -> dict[RetrieverName, object]:
        return {
            name: self.retriever(name)
            for name in RETRIEVER_NAMES
        }


def build_retrieval_components(
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
) -> RetrievalComponents:
    """Read the corpus and build the shared retrievers once."""

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

    return RetrievalComponents(
        analyzer=analyzer,
        bm25=bm25,
        dense=dense,
    )
