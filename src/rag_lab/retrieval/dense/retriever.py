from __future__ import annotations

from time import perf_counter

from rag_lab.contracts import (
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.embeddings.provider import (
    EmbeddingProvider,
)
from rag_lab.vector_store.provider import (
    VectorStore,
)


class DenseRetriever:
    """Retrieve ranked KnowledgeChunk objects from a vector store."""

    RETRIEVER_NAME = "dense"

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        if provider.dimensions != store.dimensions:
            raise ValueError(
                "embedding provider dimensions must match "
                "vector store dimensions"
            )

        self._provider = provider
        self._store = store
        self._index_version = (
            f"{self.RETRIEVER_NAME}-v1:"
            f"{store.collection_name}:"
            f"{provider.embedding_version}"
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        started_at = perf_counter()

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError("top_k must be an integer")

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        if (
            filters is not None
            and not isinstance(filters, SearchFilters)
        ):
            raise TypeError(
                "filters must be SearchFilters"
            )

        query_vector = self._provider.embed_query(query)
        candidate_count = self._store.count(
            filters=filters
        )
        matches = self._store.search(
            query_vector,
            top_k=top_k,
            filters=filters,
        )

        hits = [
            SearchHit(
                chunk=match.chunk,
                score=match.score,
                rank=rank,
                retriever=self.RETRIEVER_NAME,
            )
            for rank, match in enumerate(
                matches,
                start=1,
            )
        ]

        elapsed_ms = max(
            (perf_counter() - started_at) * 1000,
            0.0,
        )

        return SearchResult(
            query=query,
            hits=hits,
            candidate_count=candidate_count,
            elapsed_ms=elapsed_ms,
            retriever=self.RETRIEVER_NAME,
            index_version=self._index_version,
        )

    def validate_readiness(self) -> int:
        """Confirm the configured vector collection is readable.

        This intentionally avoids generating an embedding or performing a
        vector search, so a health probe does not consume model capacity.
        ``count`` verifies that the Qdrant collection exists and remains
        dimensionally compatible.
        """

        return self._store.count()
