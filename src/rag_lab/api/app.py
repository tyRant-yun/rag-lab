from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
)
from rag_lab.embeddings import (
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)
from rag_lab.retrieval.factory import (
    RetrieverName,
    build_retrieval_components,
)
from rag_lab.retrieval.hybrid import (
    HybridRetriever,
)
from rag_lab.retrieval.rerank import (
    RerankedRetriever,
)
from rag_lab.vector_store import (
    QdrantVectorStoreError,
)
from rag_lab.vector_store.cli import (
    build_qdrant_store,
)


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
    include_source_path: bool = False
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


class SearchHitResponse(BaseModel):
    """One sanitized hit in an API search response."""

    model_config = ConfigDict(
        extra="forbid",
    )

    rank: int
    score: float
    retriever: str
    chunk_id: str
    content: str
    heading_path: list[str]
    page_start: int
    page_end: int
    source_path: str | None = None


class SearchResponse(BaseModel):
    """API search response without internal contract leakage."""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str
    hits: list[SearchHitResponse]
    candidate_count: int
    elapsed_ms: float
    retriever: str
    index_version: str


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
        response_model=SearchResponse,
    )
    def search(
        payload: SearchRequest,
    ) -> SearchResponse:
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
            retriever = components.retriever(
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

            result = retriever.search(
                payload.query,
                top_k=payload.top_k,
                filters=filters,
            )

            return SearchResponse(
                query=result.query,
                hits=[
                    SearchHitResponse(
                        rank=hit.rank,
                        score=hit.score,
                        retriever=hit.retriever,
                        chunk_id=hit.chunk.chunk_id,
                        content=hit.chunk.content,
                        heading_path=list(
                            hit.chunk.heading_path
                        ),
                        page_start=hit.chunk.page_start,
                        page_end=hit.chunk.page_end,
                        source_path=(
                            hit.chunk.source_path
                            if payload.include_source_path
                            else None
                        ),
                    )
                    for hit in result.hits
                ],
                candidate_count=(
                    result.candidate_count
                ),
                elapsed_ms=result.elapsed_ms,
                retriever=result.retriever,
                index_version=result.index_version,
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
