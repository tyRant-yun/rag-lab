from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from rag_lab.contracts import (
    SearchFilters,
    SearchResult,
)
from rag_lab.retrieval.rerank.protocol import (
    Reranker,
)


class RerankedRetriever:
    """Run a base retriever with a wider window, then rerank to top_k.

    ``elapsed_ms`` in the returned result covers the complete operation:
    fetching candidates from the base retriever and reranking them.
    """

    DEFAULT_FETCH_K = 20

    def __init__(
        self,
        *,
        retriever,
        reranker: Reranker,
        fetch_k: int = DEFAULT_FETCH_K,
        clock: Callable[[], float] = perf_counter,
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

        if not callable(clock):
            raise TypeError("clock must be callable")

        self._retriever = retriever
        self._reranker = reranker
        self._fetch_k = fetch_k
        self._clock = clock

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        started_at = self._clock()
        base_result = self._retriever.search(
            query,
            top_k=self._fetch_k,
            filters=filters,
        )

        reranked_result = self._reranker.rerank(
            base_result,
            top_k=top_k,
        )

        elapsed_ms = max(
            (self._clock() - started_at) * 1000,
            0.0,
        )

        return SearchResult(
            query=reranked_result.query,
            hits=reranked_result.hits,
            candidate_count=(
                reranked_result.candidate_count
            ),
            elapsed_ms=elapsed_ms,
            retriever=reranked_result.retriever,
            index_version=reranked_result.index_version,
        )
