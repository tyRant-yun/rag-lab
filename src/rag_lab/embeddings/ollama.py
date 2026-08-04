from __future__ import annotations

import math
from collections.abc import Sequence
from hashlib import sha256
from time import perf_counter
from typing import Protocol

from httpx import RequestError as HttpxRequestError
from ollama import (
    Client,
    ResponseError,
)
from pydantic import ValidationError

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
)


class OllamaEmbeddingError(RuntimeError):
    """Stable application error for Ollama failures."""


class _EmbeddingResponse(Protocol):
    @property
    def embeddings(
        self,
    ) -> Sequence[Sequence[float]]:
        ...


class _OllamaClient(Protocol):
    def embed(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        truncate: bool,
        dimensions: int,
    ) -> _EmbeddingResponse:
        ...


class OllamaEmbeddingProvider:
    """Generate validated embeddings through Ollama."""

    PROVIDER_NAME = "ollama"
    DEFAULT_MODEL = "qwen3-embedding:0.6b"
    DEFAULT_DIMENSIONS = 1024
    DEFAULT_HOST = "http://localhost:11434"

    DEFAULT_QUERY_INSTRUCTION = (
        "Given a technical knowledge-base query, "
        "retrieve relevant passages that answer "
        "the query"
    )

    QUERY_FORMAT_VERSION = "query-v1"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        host: str = DEFAULT_HOST,
        query_instruction: str = (
            DEFAULT_QUERY_INSTRUCTION
        ),
        timeout_seconds: float = 60.0,
        client: _OllamaClient | None = None,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string"
            )

        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
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

        if not isinstance(host, str):
            raise TypeError(
                "host must be a string"
            )

        if not host.strip():
            raise ValueError(
                "host cannot be empty"
            )

        if not isinstance(
            query_instruction,
            str,
        ):
            raise TypeError(
                "query_instruction must be a string"
            )

        if not query_instruction.strip():
            raise ValueError(
                "query_instruction cannot be empty"
            )

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(
                timeout_seconds,
                (int, float),
            )
        ):
            raise TypeError(
                "timeout_seconds must be a number"
            )

        normalized_timeout = float(
            timeout_seconds
        )

        if (
            not math.isfinite(normalized_timeout)
            or normalized_timeout <= 0
        ):
            raise ValueError(
                "timeout_seconds must be finite "
                "and greater than zero"
            )

        self._model_name = model_name.strip()
        self._dimensions = dimensions
        self._host = host.strip()
        self._query_instruction = (
            query_instruction.strip()
        )
        self._timeout_seconds = (
            normalized_timeout
        )

        instruction_digest = sha256(
            self._query_instruction.encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        self._embedding_version = (
            f"{self.PROVIDER_NAME}:"
            f"{self._model_name}:"
            f"dimensions-{self._dimensions}:"
            f"{self.QUERY_FORMAT_VERSION}-"
            f"{instruction_digest}"
        )

        self._client: _OllamaClient = (
            client
            if client is not None
            else Client(
                host=self._host,
                timeout=self._timeout_seconds,
            )
        )

    @property
    def provider_name(self) -> str:
        return self.PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def embedding_version(self) -> str:
        return self._embedding_version

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        active_texts = self._validate_documents(
            texts
        )

        vectors, elapsed_ms = (
            self._request_vectors(active_texts)
        )

        return EmbeddingBatch(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            vectors=vectors,
            input_count=len(active_texts),
            elapsed_ms=elapsed_ms,
            embedding_version=(
                self.embedding_version
            ),
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        if not isinstance(text, str):
            raise TypeError(
                "query text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "query text cannot be empty"
            )

        formatted_query = (
            f"Instruct: "
            f"{self._query_instruction}\n"
            f"Query: {text}"
        )

        vectors, _ = self._request_vectors(
            [formatted_query]
        )

        return vectors[0]

    @staticmethod
    def _validate_documents(
        texts: Sequence[str],
    ) -> list[str]:
        if isinstance(texts, str):
            raise TypeError(
                "texts must be a sequence of strings, "
                "not a single string"
            )

        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a sequence of strings"
            )

        active_texts = list(texts)

        if not active_texts:
            raise ValueError(
                "texts cannot be empty"
            )

        for text in active_texts:
            if not isinstance(text, str):
                raise TypeError(
                    "every document text must "
                    "be a string"
                )

            if not text.strip():
                raise ValueError(
                    "document texts cannot be empty"
                )

        return active_texts

    def _request_vectors(
        self,
        inputs: list[str],
    ) -> tuple[list[EmbeddingVector], float]:
        started_at = perf_counter()

        try:
            response = self._client.embed(
                model=self.model_name,
                input=inputs,
                truncate=False,
                dimensions=self.dimensions,
            )
        except (
            ConnectionError,
            HttpxRequestError,
            ResponseError,
        ) as exc:
            raise OllamaEmbeddingError(
                "Ollama embedding request failed: "
                f"{exc}"
            ) from exc

        raw_embeddings = list(
            response.embeddings
        )

        if len(raw_embeddings) != len(inputs):
            raise OllamaEmbeddingError(
                "Ollama returned an unexpected "
                "number of embeddings"
            )

        vectors: list[EmbeddingVector] = []

        try:
            for values in raw_embeddings:
                vectors.append(
                    EmbeddingVector(
                        values=list(values),
                        dimensions=self.dimensions,
                    )
                )
        except (
            TypeError,
            ValidationError,
        ) as exc:
            raise OllamaEmbeddingError(
                "Ollama returned an invalid "
                "embedding vector"
            ) from exc

        elapsed_ms = max(
            (
                perf_counter()
                - started_at
            )
            * 1000,
            0.0,
        )

        return vectors, elapsed_ms
