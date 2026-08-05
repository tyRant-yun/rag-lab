from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag_lab.contracts import EmbeddingVector
from rag_lab.vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreError,
)


@pytest.mark.parametrize(
    "operation",
    ["count", "search"],
)
def test_read_operations_do_not_create_missing_collection(
    operation: str,
) -> None:
    client = QdrantClient(":memory:")
    collection_name = f"missing-{operation}"
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        dimensions=2,
    )

    with pytest.raises(
        QdrantVectorStoreError,
        match="does not exist",
    ):
        if operation == "count":
            store.count()
        else:
            store.search(
                EmbeddingVector(
                    values=[0.5, 0.5],
                    dimensions=2,
                )
            )

    assert client.collection_exists(collection_name) is False


def test_read_operations_use_existing_compatible_collection() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="existing-read-only",
        dimensions=2,
    )
    store.ensure_collection()

    assert store.count() == 0
    assert store.search(
        EmbeddingVector(
            values=[0.5, 0.5],
            dimensions=2,
        )
    ) == []
