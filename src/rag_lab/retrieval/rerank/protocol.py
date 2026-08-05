from __future__ import annotations

from typing import Protocol

from rag_lab.contracts import SearchResult


class Reranker(Protocol):
    """Re-rank a SearchResult and return a smaller ranked result."""

    def rerank(
        self,
        result: SearchResult,
        *,
        top_k: int,
    ) -> SearchResult:
        ...
