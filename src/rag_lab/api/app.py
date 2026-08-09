from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
from rag_lab.knowledge_base import (
    PublicKnowledgeBaseInfo,
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


class PublicSearchRequest(BaseModel):
    """Stable browser-facing search contract."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class PublicCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    section: str
    pages: str


class PublicSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    citation: PublicCitation


class PublicSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    results: list[PublicSearchResult]


_PUBLIC_KNOWLEDGE_BASE = PublicKnowledgeBaseInfo(
    title="计算机网络基础知识库",
    coverage="《计算机网络：自顶向下方法》第 8 版的全部可读文本（第 5–501 页）",
    topics=["应用层", "运输层", "网络层", "链路层与局域网", "无线网络与网络安全"],
    capabilities=["定位概念的原文依据", "按章节和页码复习", "比较相关网络基础主题"],
    guidance=["一次提出一个明确问题", "优先使用教材中的概念词", "结合章节和页码核对原文"],
    limitations=["当前仅覆盖这一本教材的可读文本", "结果是可核对的检索依据，不代替完整教材阅读"],
)


def _apply_security_headers(response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'"
    )


def _check_ollama_ready(*, host: str, model: str, timeout: float) -> None:
    """Check the configured Ollama endpoint without generating embeddings."""

    response = httpx.get(
        f"{host.rstrip('/')}/api/tags",
        timeout=min(timeout, 5.0),
    )
    response.raise_for_status()
    payload = response.json()
    available_models = {
        item.get("name")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    if model not in available_models:
        raise RuntimeError("configured Ollama model is unavailable")


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
    enable_debug_routes: bool = True,
    knowledge_base_info: PublicKnowledgeBaseInfo | None = None,
    readiness_checker: Callable[[], None] | None = None,
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

    def default_readiness_check() -> None:
        components.dense.validate_readiness()
        _check_ollama_ready(
            host=host,
            model=model,
            timeout=embedding_timeout_seconds,
        )

    active_readiness_checker = (
        readiness_checker
        if readiness_checker is not None
        else default_readiness_check
    )

    static_directory = Path(__file__).with_name("static")
    knowledge_base_metadata = (
        knowledge_base_info or _PUBLIC_KNOWLEDGE_BASE
    )
    app = FastAPI(
        title="RAG Lab Retrieval API",
        description=(
            "Expose BM25, Dense, Hybrid (RRF) and "
            "reranked retrieval over a KnowledgeChunk "
            "corpus."
        ),
        version="0.1.0",
        docs_url="/docs" if enable_debug_routes else None,
        openapi_url="/openapi.json" if enable_debug_routes else None,
    )
    app.mount("/static", StaticFiles(directory=static_directory), name="static")

    @app.middleware("http")
    async def public_boundary(request: Request, call_next):
        is_public_search = request.url.path == "/api/v1/search"
        content_length = request.headers.get("content-length")
        if (
            is_public_search
            and content_length is not None
            and content_length.isdecimal()
            and int(content_length) > 8192
        ):
            response = JSONResponse(
                status_code=413,
                content={
                    "code": "request_too_large",
                    "message": "请求内容过大。",
                    "request_id": f"req_{uuid4().hex}",
                },
            )
            _apply_security_headers(response)
            return response
        if is_public_search and len(await request.body()) > 8192:
            response = JSONResponse(
                status_code=413,
                content={
                    "code": "request_too_large",
                    "message": "请求内容过大。",
                    "request_id": f"req_{uuid4().hex}",
                },
            )
            _apply_security_headers(response)
            return response
        try:
            response = await call_next(request)
        except Exception:
            if is_public_search:
                response = JSONResponse(
                    status_code=503,
                    content={
                        "code": "search_unavailable",
                        "message": "服务暂时不可用，请稍后重试。",
                        "request_id": f"req_{uuid4().hex}",
                    },
                )
                _apply_security_headers(response)
                return response
            raise
        _apply_security_headers(response)
        return response

    @app.exception_handler(RequestValidationError)
    async def public_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        if request.url.path == "/api/v1/search":
            return JSONResponse(
                status_code=422,
                content={
                    "code": "invalid_request",
                    "message": "请求格式不正确。",
                    "request_id": f"req_{uuid4().hex}",
                },
            )
        return await request_validation_exception_handler(request, exc)

    @app.get("/", include_in_schema=False)
    def web_mvp() -> FileResponse:
        return FileResponse(static_directory / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> dict[str, str]:
        try:
            active_readiness_checker()
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready"},
            )
        return {"status": "ready"}

    @app.get(
        "/api/v1/knowledge-base",
        response_model=PublicKnowledgeBaseInfo,
    )
    def public_knowledge_base() -> PublicKnowledgeBaseInfo:
        return knowledge_base_metadata

    @app.post("/api/v1/search", response_model=PublicSearchResponse)
    def public_search(payload: PublicSearchRequest) -> PublicSearchResponse:
        request_id = f"req_{uuid4().hex}"
        try:
            result = components.retriever("rerank").search(
                payload.query,
                top_k=5,
                filters=None,
            )
        except (TypeError, ValueError, OllamaEmbeddingError, QdrantVectorStoreError):
            return JSONResponse(
                status_code=503,
                content={
                    "code": "search_unavailable",
                    "message": "服务暂时不可用，请稍后重试。",
                    "request_id": request_id,
                },
            )
        return PublicSearchResponse(
            request_id=request_id,
            results=[
                PublicSearchResult(
                    content=hit.chunk.content,
                    citation=PublicCitation(
                        title=hit.chunk.heading_path[0],
                        section=hit.chunk.heading_path[-1],
                        pages=(
                            str(hit.chunk.page_start)
                            if hit.chunk.page_start == hit.chunk.page_end
                            else f"{hit.chunk.page_start}-{hit.chunk.page_end}"
                        ),
                    ),
                )
                for hit in result.hits
            ],
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

    if enable_debug_routes:
        app.post("/search", response_model=SearchResponse)(search)

    return app
