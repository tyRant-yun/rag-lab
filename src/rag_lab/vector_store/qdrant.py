from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from qdrant_client import (
    QdrantClient,
    models,
)

from rag_lab.contracts import (
    EmbeddingVector,
    SearchFilters,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.vector_store.payload import (
    chunk_point_id,
    filters_to_qdrant_filter,
    payload_to_chunk,
    record_to_payload,
)


class QdrantVectorStoreError(RuntimeError):
    """Base error for stable Qdrant failures."""


class QdrantCollectionConfigurationError(
    QdrantVectorStoreError
):
    """Raised when an existing collection is incompatible."""


class QdrantVectorStore:
    """Qdrant implementation of the VectorStore contract."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        dimensions: int,
    ) -> None:
        if not isinstance(collection_name, str):
            raise TypeError(
                "collection_name must be a string"
            )

        if not collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty"
            )

        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
        ):
            raise TypeError(
                "dimensions must be an integer"
            )

        if dimensions < 1:
            raise ValueError(
                "dimensions must be at least 1"
            )

        self._client = client
        self._collection_name = collection_name
        self._dimensions = dimensions

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def ensure_collection(self) -> None:
        try:
            collection_exists = (
                self._client.collection_exists(
                    self.collection_name
                )
            )
        except Exception as error:
            raise QdrantVectorStoreError(
                "failed to inspect Qdrant collection "
                f"{self.collection_name!r}: {error}"
            ) from error

        if not collection_exists:
            try:
                self._client.create_collection(
                    collection_name=(
                        self.collection_name
                    ),
                    vectors_config=(
                        models.VectorParams(
                            size=self.dimensions,
                            distance=(
                                models.Distance.COSINE
                            ),
                        )
                    ),
                )
            except Exception as error:
                raise QdrantVectorStoreError(
                    "failed to create Qdrant collection "
                    f"{self.collection_name!r}: {error}"
                ) from error

            return

        try:
            collection_info = (
                self._client.get_collection(
                    self.collection_name
                )
            )
        except Exception as error:
            raise QdrantVectorStoreError(
                "failed to read Qdrant collection "
                f"{self.collection_name!r}: {error}"
            ) from error

        vectors_config = (
            collection_info
            .config
            .params
            .vectors
        )

        if not isinstance(
            vectors_config,
            models.VectorParams,
        ):
            raise QdrantCollectionConfigurationError(
                "Qdrant collection must contain one "
                "unnamed dense vector"
            )

        if vectors_config.size != self.dimensions:
            raise QdrantCollectionConfigurationError(
                "Qdrant collection dimensions "
                f"mismatch: expected {self.dimensions}, "
                f"found {vectors_config.size}"
            )

        if (
            vectors_config.distance
            != models.Distance.COSINE
        ):
            raise QdrantCollectionConfigurationError(
                "Qdrant collection distance "
                "must be Cosine"
            )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        if isinstance(records, (str, bytes)):
            raise TypeError(
                "records must be a sequence "
                "of VectorRecord"
            )

        active_records = list(records)

        if not active_records:
            raise ValueError(
                "records cannot be empty"
            )

        if any(
            not isinstance(record, VectorRecord)
            for record in active_records
        ):
            raise TypeError(
                "records must contain only "
                "VectorRecord values"
            )

        if any(
            record.vector.dimensions
            != self.dimensions
            for record in active_records
        ):
            raise ValueError(
                "record vector dimensions must "
                "match store dimensions"
            )

        embedding_versions = {
            record.embedding_version
            for record in active_records
        }

        if len(embedding_versions) != 1:
            raise ValueError(
                "records must use one "
                "embedding_version"
            )

        self.ensure_collection()

        points = [
            models.PointStruct(
                id=chunk_point_id(
                    record.chunk.chunk_id
                ),
                vector=record.vector.values,
                payload=record_to_payload(record),
            )
            for record in active_records
        ]

        started_at = perf_counter()

        try:
            self._client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )
        except Exception as error:
            raise QdrantVectorStoreError(
                "failed to upsert Qdrant points "
                f"into {self.collection_name!r}: "
                f"{error}"
            ) from error

        elapsed_ms = max(
            (
                perf_counter()
                - started_at
            )
            * 1000,
            0.0,
        )

        embedding_version = next(
            iter(embedding_versions)
        )

        return VectorWriteReport(
            collection_name=self.collection_name,
            dimensions=self.dimensions,
            input_count=len(active_records),
            upserted_count=len(active_records),
            elapsed_ms=elapsed_ms,
            embedding_version=embedding_version,
        )

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        if not isinstance(
            vector,
            EmbeddingVector,
        ):
            raise TypeError(
                "vector must be EmbeddingVector"
            )

        if vector.dimensions != self.dimensions:
            raise ValueError(
                "query vector dimensions must "
                "match store dimensions"
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

        if (
            filters is not None
            and not isinstance(
                filters,
                SearchFilters,
            )
        ):
            raise TypeError(
                "filters must be SearchFilters"
            )

        self.ensure_collection()

        try:
            response = self._client.query_points(
                collection_name=(
                    self.collection_name
                ),
                query=vector.values,
                query_filter=(
                    filters_to_qdrant_filter(
                        filters
                    )
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:
            raise QdrantVectorStoreError(
                "failed to query Qdrant collection "
                f"{self.collection_name!r}: {error}"
            ) from error

        matches: list[VectorMatch] = []

        for point in response.points:
            if point.payload is None:
                raise QdrantVectorStoreError(
                    "Qdrant result is missing payload"
                )

            try:
                chunk = payload_to_chunk(
                    point.payload
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise QdrantVectorStoreError(
                    "Qdrant result contains an "
                    "invalid KnowledgeChunk payload"
                ) from error

            matches.append(
                VectorMatch(
                    chunk=chunk,
                    score=float(point.score),
                )
            )

        return matches
