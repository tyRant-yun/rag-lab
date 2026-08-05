from __future__ import annotations

import math

from rag_lab.contracts import (
    SearchHit,
    SearchResult,
)
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)
from rag_lab.retrieval.rerank.protocol import (
    Reranker,
)


class LexicalOverlapReranker:
    """Deterministic reranker over fused query-term overlap.

    Each candidate is re-scored as::

        score = rrf_weight * fused_score
              + overlap_weight * query_term_overlap
              + heading_weight * heading_term_overlap

    ``query_term_overlap`` is the fraction of query terms that also appear
    in the chunk ``index_text``; ``heading_term_overlap`` is the same
    fraction measured against the chunk ``heading_path``.  Ties break by
    ascending chunk ID.  The rerank is fully deterministic and needs no
    model call.
    """

    RETRIEVER_SUFFIX = "+rerank"
    INDEX_VERSION_SUFFIX = "|rerank-v1:lexical-overlap"

    def __init__(
        self,
        *,
        analyzer: LexicalAnalyzer,
        rrf_weight: float = 1.0,
        overlap_weight: float = 1.0,
        heading_weight: float = 1.0,
    ) -> None:
        if not isinstance(analyzer, LexicalAnalyzer):
            raise TypeError(
                "analyzer must be LexicalAnalyzer"
            )

        self._analyzer = analyzer
        self._rrf_weight = self._validate_weight(
            "rrf_weight",
            rrf_weight,
        )
        self._overlap_weight = self._validate_weight(
            "overlap_weight",
            overlap_weight,
        )
        self._heading_weight = self._validate_weight(
            "heading_weight",
            heading_weight,
        )

    @staticmethod
    def _validate_weight(
        name: str,
        value: float,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a number"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be a number"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0:
            raise ValueError(
                f"{name} must not be negative"
            )

        return float(value)

    def rerank(
        self,
        result: SearchResult,
        *,
        top_k: int,
    ) -> SearchResult:
        if not isinstance(result, SearchResult):
            raise TypeError(
                "result must be SearchResult"
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

        query_terms = frozenset(
            self._analyzer.analyze(result.query)
        )

        if not query_terms:
            return result

        scored: list[
            tuple[
                float,
                str,
                SearchHit,
            ]
        ] = []

        for hit in result.hits:
            chunk = hit.chunk
            chunk_terms = frozenset(
                self._analyzer.analyze(
                    chunk.index_text
                )
            )
            heading_terms = frozenset(
                self._analyzer.analyze(
                    " ".join(chunk.heading_path)
                )
            )

            overlap = (
                len(query_terms & chunk_terms)
                / len(query_terms)
            )
            heading_overlap = (
                len(query_terms & heading_terms)
                / len(query_terms)
            )
            score = (
                self._rrf_weight * hit.score
                + self._overlap_weight * overlap
                + self._heading_weight
                * heading_overlap
            )

            scored.append(
                (
                    score,
                    chunk.chunk_id,
                    hit,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        selected = scored[:top_k]
        retriever = (
            result.retriever
            + self.RETRIEVER_SUFFIX
        )

        hits = [
            SearchHit(
                chunk=hit.chunk,
                score=score,
                rank=rank,
                retriever=retriever,
            )
            for rank, (
                score,
                _,
                hit,
            ) in enumerate(
                selected,
                start=1,
            )
        ]

        return SearchResult(
            query=result.query,
            hits=hits,
            candidate_count=result.candidate_count,
            elapsed_ms=result.elapsed_ms,
            retriever=retriever,
            index_version=(
                result.index_version
                + self.INDEX_VERSION_SUFFIX
            ),
        )
