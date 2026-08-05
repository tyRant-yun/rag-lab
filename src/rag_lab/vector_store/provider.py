from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Protocol,
    runtime_checkable,
)

from rag_lab.contracts import (
    EmbeddingVector,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)


@runtime_checkable
class VectorStore(Protocol):
    """Storage-neutral vector database contract."""

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the configured vector dimensions."""
        ...

    def ensure_collection(self) -> None:
        """Create or validate the vector collection."""
        ...

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        """Insert or replace vector records."""
        ...

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        """Count records matching storage-neutral filters."""
        ...

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        """Return vector matches ordered by similarity."""
        ...
