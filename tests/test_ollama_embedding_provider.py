from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from httpx import ReadTimeout
from ollama import ResponseError

from rag_lab.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingError,
    OllamaEmbeddingProvider,
)


@dataclass(frozen=True)
class FakeEmbeddingResponse:
    embeddings: list[list[float]]


class FakeOllamaClient:
    def __init__(
        self,
        embeddings: list[list[float]],
    ) -> None:
        self._response = (
            FakeEmbeddingResponse(
                embeddings=embeddings,
            )
        )
        self.calls: list[
            dict[str, object]
        ] = []

    def embed(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        truncate: bool,
        dimensions: int,
    ) -> FakeEmbeddingResponse:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "truncate": truncate,
                "dimensions": dimensions,
            }
        )

        return self._response


class FailingOllamaClient:
    def __init__(
        self,
        error: Exception,
    ) -> None:
        self._error = error

    def embed(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        truncate: bool,
        dimensions: int,
    ) -> FakeEmbeddingResponse:
        raise self._error


def make_provider(
    client: (
        FakeOllamaClient
        | FailingOllamaClient
    ),
) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        model_name="test-model",
        dimensions=3,
        host="http://localhost:11434",
        query_instruction=(
            "Retrieve technical passages "
            "that answer the query"
        ),
        client=client,
    )


def test_provider_matches_embedding_protocol():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    assert isinstance(
        provider,
        EmbeddingProvider,
    )


def test_provider_exposes_identity():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    assert provider.provider_name == "ollama"
    assert provider.model_name == "test-model"
    assert provider.dimensions == 3

    assert provider.embedding_version.startswith(
        "ollama:test-model:"
        "dimensions-3:query-v1-"
    )


def test_documents_are_embedded_as_batch():
    client = FakeOllamaClient(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    provider = make_provider(client)

    result = provider.embed_documents(
        [
            "第一段文档",
            "第二段文档",
        ]
    )

    assert result.input_count == 2
    assert result.dimensions == 3
    assert len(result.vectors) == 2

    assert client.calls == [
        {
            "model": "test-model",
            "input": [
                "第一段文档",
                "第二段文档",
            ],
            "truncate": False,
            "dimensions": 3,
        }
    ]


def test_document_text_has_no_query_instruction():
    client = FakeOllamaClient(
        [[0.1, 0.2, 0.3]]
    )
    provider = make_provider(client)

    provider.embed_documents(
        ["网络协议定义通信规则。"]
    )

    sent_input = client.calls[0]["input"]

    assert sent_input == [
        "网络协议定义通信规则。"
    ]


def test_query_receives_instruction():
    client = FakeOllamaClient(
        [[0.1, 0.2, 0.3]]
    )
    provider = make_provider(client)

    result = provider.embed_query(
        "什么是网络协议？"
    )

    assert result.values == [
        0.1,
        0.2,
        0.3,
    ]

    assert client.calls[0]["input"] == [
        "Instruct: Retrieve technical passages "
        "that answer the query\n"
        "Query: 什么是网络协议？"
    ]


def test_documents_cannot_be_single_string():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    with pytest.raises(
        TypeError,
        match="not a single string",
    ):
        provider.embed_documents(
            "不是文档列表"
        )


def test_documents_cannot_be_empty():
    provider = make_provider(
        FakeOllamaClient([])
    )

    with pytest.raises(
        ValueError,
        match="texts cannot be empty",
    ):
        provider.embed_documents([])


def test_document_entries_cannot_be_empty():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    with pytest.raises(
        ValueError,
        match="document texts cannot be empty",
    ):
        provider.embed_documents([" "])


def test_query_cannot_be_empty():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    with pytest.raises(
        ValueError,
        match="query text cannot be empty",
    ):
        provider.embed_query(" ")


def test_response_count_must_match_input_count():
    provider = make_provider(
        FakeOllamaClient(
            [[0.1, 0.2, 0.3]]
        )
    )

    with pytest.raises(
        OllamaEmbeddingError,
        match=(
            "unexpected number of embeddings"
        ),
    ):
        provider.embed_documents(
            [
                "第一段",
                "第二段",
            ]
        )


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [0.1, 0.2],
        [0.0, 0.0, 0.0],
        [0.1, math.nan, 0.3],
        [0.1, math.inf, 0.3],
    ],
)
def test_invalid_vectors_are_rejected(
    invalid_vector: list[float],
):
    provider = make_provider(
        FakeOllamaClient(
            [invalid_vector]
        )
    )

    with pytest.raises(
        OllamaEmbeddingError,
        match="invalid embedding vector",
    ):
        provider.embed_documents(
            ["文档"]
        )


def test_connection_errors_are_wrapped():
    provider = make_provider(
        FailingOllamaClient(
            ConnectionError(
                "connection refused"
            )
        )
    )

    with pytest.raises(
        OllamaEmbeddingError,
        match="Ollama embedding request failed",
    ):
        provider.embed_query(
            "测试查询"
        )


def test_timeout_errors_are_wrapped():
    provider = make_provider(
        FailingOllamaClient(
            ReadTimeout(
                "request timed out"
            )
        )
    )

    with pytest.raises(
        OllamaEmbeddingError,
        match="Ollama embedding request failed",
    ):
        provider.embed_query(
            "测试查询"
        )


def test_response_errors_are_wrapped():
    provider = make_provider(
        FailingOllamaClient(
            ResponseError(
                "model not found",
                status_code=404,
            )
        )
    )

    with pytest.raises(
        OllamaEmbeddingError,
        match="Ollama embedding request failed",
    ):
        provider.embed_query(
            "测试查询"
        )


@pytest.mark.parametrize(
    "dimensions",
    [
        0,
        -1,
    ],
)
def test_dimensions_must_be_positive(
    dimensions: int,
):
    with pytest.raises(
        ValueError,
        match="dimensions must be at least 1",
    ):
        OllamaEmbeddingProvider(
            dimensions=dimensions,
            client=FakeOllamaClient([]),
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_timeout_must_be_valid(
    timeout_seconds: float,
):
    with pytest.raises(
        ValueError,
        match=(
            "timeout_seconds must be finite "
            "and greater than zero"
        ),
    ):
        OllamaEmbeddingProvider(
            timeout_seconds=timeout_seconds,
            client=FakeOllamaClient([]),
        )
