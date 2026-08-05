from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
)


class HybridRetriever:
    """Fuse BM25 and dense rankings with Reciprocal Rank Fusion.

    Each underlying retriever is asked for its top ``per_retriever_k`` hits.
    A chunk receives one ``1 / (rrf_k + rank)`` contribution per retriever
    that returns it.  The fused list is deterministic: descending fused
    score, then ascending chunk ID.

    ``candidate_count`` is the number of distinct chunks seen in the fused
    per-retriever lists (the union), not the total filtered corpus size.
    """

    RETRIEVER_NAME = "hybrid"
    DEFAULT_RRF_K = 60
    DEFAULT_PER_RETRIEVER_K = 10

    def __init__(
        self,
        *,
        bm25,
        dense,
        rrf_k: int = DEFAULT_RRF_K,
        per_retriever_k: int = DEFAULT_PER_RETRIEVER_K,
    ) -> None:
        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
        ):
            raise TypeError(
                "rrf_k must be an integer"
            )

        if rrf_k < 1:
            raise ValueError(
                "rrf_k must be at least 1"
            )

        if (
            isinstance(per_retriever_k, bool)
            or not isinstance(per_retriever_k, int)
        ):
            raise TypeError(
                "per_retriever_k must be an integer"
            )

        if per_retriever_k < 1:
            raise ValueError(
                "per_retriever_k must be at least 1"
            )

        self._bm25 = bm25
        self._dense = dense
        self._rrf_k = rrf_k
        self._per_retriever_k = per_retriever_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        started_at = perf_counter()

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        active_filters = (
            filters
            if filters is not None
            else SearchFilters()
        )

        if not isinstance(
            active_filters,
            SearchFilters,
        ):
            raise TypeError(
                "filters must be SearchFilters"
            )

        bm25_result = self._bm25.search(
            query,
            top_k=self._per_retriever_k,
            filters=active_filters,
        )
        dense_result = self._dense.search(
            query,
            top_k=self._per_retriever_k,
            filters=active_filters,
        )

        fused_score: dict[str, float] = (
            defaultdict(float)
        )
        chunks: dict[str, KnowledgeChunk] = {}

        for result in (bm25_result, dense_result):
            for hit in result.hits:
                chunk_id = hit.chunk.chunk_id
                fused_score[chunk_id] += (
                    1.0 / (self._rrf_k + hit.rank)
                )
                chunks[chunk_id] = hit.chunk

        ordered = sorted(
            fused_score.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        selected = ordered[:top_k]

        hits = [
            SearchHit(
                chunk=chunks[chunk_id],
                score=score,
                rank=rank,
                retriever=self.RETRIEVER_NAME,
            )
            for rank, (
                chunk_id,
                score,
            ) in enumerate(
                selected,
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
            candidate_count=len(fused_score),
            elapsed_ms=elapsed_ms,
            retriever=self.RETRIEVER_NAME,
            index_version=(
                "hybrid-v1:"
                f"{bm25_result.index_version}|"
                f"{dense_result.index_version}"
            ),
        )
