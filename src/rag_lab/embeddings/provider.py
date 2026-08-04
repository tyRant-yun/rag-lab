from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Protocol,
    runtime_checkable,
)

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Storage-neutral embedding provider contract."""

    @property
    def provider_name(self) -> str:
        """Return the embedding provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the configured vector dimensions."""
        ...

    @property
    def embedding_version(self) -> str:
        """Return the reproducible embedding version."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        """Embed document texts as one validated batch."""
        ...

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        """Embed one retrieval query."""
        ...
