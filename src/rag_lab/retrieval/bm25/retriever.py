from __future__ import annotations

from time import perf_counter

from rag_lab.contracts import (
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.retrieval.bm25.index import (
    BM25Index,
)


class BM25Retriever:
    """Retrieve ranked KnowledgeChunk objects from a BM25 index."""

    RETRIEVER_NAME = "bm25"

    def __init__(
        self,
        *,
        index: BM25Index,
    ) -> None:
        self._index = index

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

        query_terms = self._index.analyze_query(
            query
        )
        scores = self._index.score_terms(
            query_terms
        )
        query_term_set = frozenset(query_terms)

        candidates: list[
            tuple[int, KnowledgeChunk, float]
        ] = []

        for position, (chunk, score) in enumerate(
            zip(
                self._index.chunks,
                scores,
                strict=True,
            )
        ):
            if not self._matches_filters(
                chunk,
                active_filters,
            ):
                continue

            candidates.append(
                (
                    position,
                    chunk,
                    score,
                )
            )

        candidate_count = len(candidates)

        ranked_candidates = [
            candidate
            for candidate in candidates
            if not query_term_set.isdisjoint(
                self._index.tokenized_corpus[
                    candidate[0]
                ]
            )
        ]

        ranked_candidates.sort(
            key=lambda candidate: (
                -candidate[2],
                candidate[0],
            )
        )

        selected = ranked_candidates[:top_k]

        hits = [
            SearchHit(
                chunk=chunk,
                score=float(score),
                rank=rank,
                retriever=self.RETRIEVER_NAME,
            )
            for rank, (
                _,
                chunk,
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
            candidate_count=candidate_count,
            elapsed_ms=elapsed_ms,
            retriever=self.RETRIEVER_NAME,
            index_version=self._index.index_version,
        )

    @staticmethod
    def _matches_filters(
        chunk: KnowledgeChunk,
        filters: SearchFilters,
    ) -> bool:
        if (
            filters.document_ids is not None
            and chunk.document_id
            not in filters.document_ids
        ):
            return False

        if filters.heading_prefix is not None:
            prefix_length = len(
                filters.heading_prefix
            )

            if (
                chunk.heading_path[:prefix_length]
                != filters.heading_prefix
            ):
                return False

        if (
            filters.page_start is not None
            and chunk.page_end
            < filters.page_start
        ):
            return False

        if (
            filters.page_end is not None
            and chunk.page_start
            > filters.page_end
        ):
            return False

        return True
