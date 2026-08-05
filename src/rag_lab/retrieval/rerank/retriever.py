from __future__ import annotations

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.retrieval.rerank.protocol import (
    Reranker,
)


class RerankedRetriever:
    """Run a base retriever with a wider window, then rerank to top_k."""

    DEFAULT_FETCH_K = 20

    def __init__(
        self,
        *,
        retriever,
        reranker: Reranker,
        fetch_k: int = DEFAULT_FETCH_K,
    ) -> None:
        if (
            isinstance(fetch_k, bool)
            or not isinstance(fetch_k, int)
        ):
            raise TypeError(
                "fetch_k must be an integer"
            )

        if fetch_k < 1:
            raise ValueError(
                "fetch_k must be at least 1"
            )

        self._retriever = retriever
        self._reranker = reranker
        self._fetch_k = fetch_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        base_result = self._retriever.search(
            query,
            top_k=self._fetch_k,
            filters=filters,
        )

        return self._reranker.rerank(
            base_result,
            top_k=top_k,
        )
