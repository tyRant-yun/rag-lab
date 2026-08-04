from __future__ import annotations

from collections.abc import Sequence

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
)
from rag_lab.embeddings import EmbeddingProvider


class FakeEmbeddingProvider:
    """Test double without an external model server."""

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def embedding_version(self) -> str:
        return "fake:fake-embedding-model:3:v1"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        vectors = [
            EmbeddingVector(
                values=[
                    float(position),
                    0.5,
                    1.0,
                ],
                dimensions=self.dimensions,
            )
            for position, _ in enumerate(
                texts,
                start=1,
            )
        ]

        return EmbeddingBatch(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            vectors=vectors,
            input_count=len(texts),
            elapsed_ms=0.0,
            embedding_version=(
                self.embedding_version
            ),
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            values=[0.25, 0.5, 1.0],
            dimensions=self.dimensions,
        )


def test_embedding_contracts_are_publicly_exported():
    import rag_lab.contracts as contracts

    assert "EmbeddingBatch" in contracts.__all__
    assert "EmbeddingVector" in contracts.__all__


def test_embedding_provider_is_publicly_exported():
    import rag_lab.embeddings as embeddings

    assert "EmbeddingProvider" in embeddings.__all__


def test_structural_provider_matches_protocol():
    provider = FakeEmbeddingProvider()

    assert isinstance(
        provider,
        EmbeddingProvider,
    )


def test_incomplete_object_does_not_match_protocol():
    assert not isinstance(
        object(),
        EmbeddingProvider,
    )


def test_provider_exposes_identity():
    provider = FakeEmbeddingProvider()

    assert provider.provider_name == "fake"
    assert provider.model_name == (
        "fake-embedding-model"
    )
    assert provider.dimensions == 3
    assert provider.embedding_version == (
        "fake:fake-embedding-model:3:v1"
    )


def test_provider_embeds_document_batch():
    provider: EmbeddingProvider = (
        FakeEmbeddingProvider()
    )

    result = provider.embed_documents(
        [
            "第一段文档",
            "第二段文档",
        ]
    )

    assert result.provider == "fake"
    assert result.input_count == 2
    assert result.dimensions == 3
    assert len(result.vectors) == 2


def test_provider_embeds_query():
    provider: EmbeddingProvider = (
        FakeEmbeddingProvider()
    )

    result = provider.embed_query(
        "什么是网络协议？"
    )

    assert result.dimensions == 3
    assert result.values == [
        0.25,
        0.5,
        1.0,
    ]
