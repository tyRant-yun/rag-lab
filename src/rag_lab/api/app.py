from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.embeddings import (
    OllamaEmbeddingError,
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
from rag_lab.vector_store import (
    QdrantVectorStoreError,
    VectorStore,
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


class SearchRequest(BaseModel):
    """One retrieval API request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        max_length=500,
    )
    retriever: RetrieverName = "rerank"
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )
    rrf_k: int = Field(
        default=HybridRetriever.DEFAULT_RRF_K,
        ge=1,
        le=1000,
    )
    per_retriever_k: int = Field(
        default=(
            HybridRetriever.DEFAULT_PER_RETRIEVER_K
        ),
        ge=1,
        le=100,
    )
    fetch_k: int = Field(
        default=RerankedRetriever.DEFAULT_FETCH_K,
        ge=1,
        le=200,
    )
    rrf_weight: float = Field(
        default=1.0,
        ge=0,
        le=100,
    )
    overlap_weight: float = Field(
        default=1.0,
        ge=0,
        le=100,
    )
    heading_weight: float = Field(
        default=1.0,
        ge=0,
        le=100,
    )
    document_ids: list[str] | None = None
    heading_prefix: list[str] | None = None
    page_start: int | None = Field(
        default=None,
        ge=1,
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_page_range(self) -> SearchRequest:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must not precede page_start"
            )

        return self


def create_app(
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
        OllamaEmbeddingProvider.DEFAULT_DIMENSIONS
    ),
    embedding_timeout_seconds: float = 60.0,
    qdrant_timeout_seconds: int = 10,
    user_words: Sequence[str] = (),
    stopwords: Sequence[str] = (),
    provider_factory=OllamaEmbeddingProvider,
    store_factory=build_qdrant_store,
) -> FastAPI:
    """Build a retrieval API app over a chunks corpus and Qdrant."""

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
    store: VectorStore = store_factory(
        url=url,
        collection_name=collection,
        dimensions=dimensions,
        timeout_seconds=qdrant_timeout_seconds,
    )
    dense = DenseRetriever(
        provider=provider,
        store=store,
    )

    def build_retriever(
        name: RetrieverName,
        *,
        rrf_k: int,
        per_retriever_k: int,
        fetch_k: int,
        rrf_weight: float,
        overlap_weight: float,
        heading_weight: float,
    ):
        if name == "bm25":
            return bm25

        if name == "dense":
            return dense

        hybrid = HybridRetriever(
            bm25=bm25,
            dense=dense,
            rrf_k=rrf_k,
            per_retriever_k=per_retriever_k,
        )

        if name == "hybrid":
            return hybrid

        reranker = LexicalOverlapReranker(
            analyzer=analyzer,
            rrf_weight=rrf_weight,
            overlap_weight=overlap_weight,
            heading_weight=heading_weight,
        )

        return RerankedRetriever(
            retriever=hybrid,
            reranker=reranker,
            fetch_k=fetch_k,
        )

    app = FastAPI(
        title="RAG Lab Retrieval API",
        description=(
            "Expose BM25, Dense, Hybrid (RRF) and "
            "reranked retrieval over a KnowledgeChunk "
            "corpus."
        ),
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/search",
        response_model=SearchResult,
    )
    def search(
        payload: SearchRequest,
    ) -> SearchResult:
        try:
            filters = SearchFilters(
                document_ids=(
                    payload.document_ids
                ),
                heading_prefix=(
                    payload.heading_prefix
                ),
                page_start=payload.page_start,
                page_end=payload.page_end,
            )
        except (
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        try:
            retriever = build_retriever(
                payload.retriever,
                rrf_k=payload.rrf_k,
                per_retriever_k=(
                    payload.per_retriever_k
                ),
                fetch_k=payload.fetch_k,
                rrf_weight=payload.rrf_weight,
                overlap_weight=(
                    payload.overlap_weight
                ),
                heading_weight=(
                    payload.heading_weight
                ),
            )

            return retriever.search(
                payload.query,
                top_k=payload.top_k,
                filters=filters,
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error
        except (
            OllamaEmbeddingError,
            QdrantVectorStoreError,
        ) as error:
            raise HTTPException(
                status_code=502,
                detail=str(error),
            ) from error

    return app
