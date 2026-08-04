import math

import pytest
from pydantic import ValidationError

from rag_lab.contracts.embeddings import (
    EmbeddingBatch,
    EmbeddingRunReport,
    EmbeddingVector,
)


def make_vector() -> EmbeddingVector:
    return EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        dimensions=3,
    )


def make_batch_data() -> dict[str, object]:
    return {
        "provider": "ollama",
        "model": "qwen3-embedding:0.6b",
        "dimensions": 3,
        "vectors": [
            make_vector(),
            EmbeddingVector(
                values=[0.4, 0.5, 0.6],
                dimensions=3,
            ),
        ],
        "input_count": 2,
        "elapsed_ms": 12.5,
        "embedding_version": (
            "ollama:qwen3-embedding:0.6b:"
            "dimensions-3:v1"
        ),
    }


def test_embedding_vector_serializes_contract():
    vector = make_vector()

    assert vector.to_dict() == {
        "values": [0.1, 0.2, 0.3],
        "dimensions": 3,
    }


def test_embedding_vector_cannot_be_empty():
    with pytest.raises(
        ValidationError,
        match="values cannot be empty",
    ):
        EmbeddingVector(
            values=[],
            dimensions=3,
        )


def test_embedding_vector_dimensions_must_be_positive():
    with pytest.raises(
        ValidationError,
        match="dimensions must be at least 1",
    ):
        EmbeddingVector(
            values=[0.1],
            dimensions=0,
        )


def test_embedding_vector_length_must_match_dimensions():
    with pytest.raises(
        ValidationError,
        match="values length must equal dimensions",
    ):
        EmbeddingVector(
            values=[0.1, 0.2],
            dimensions=3,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_embedding_vector_values_must_be_finite(
    invalid_value: float,
):
    with pytest.raises(
        ValidationError,
        match=(
            "values must contain only finite numbers"
        ),
    ):
        EmbeddingVector(
            values=[0.1, invalid_value],
            dimensions=2,
        )


def test_embedding_vector_cannot_be_all_zero():
    with pytest.raises(
        ValidationError,
        match="values cannot be an all-zero vector",
    ):
        EmbeddingVector(
            values=[0.0, 0.0, -0.0],
            dimensions=3,
        )


def test_embedding_vector_is_frozen():
    vector = make_vector()

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        vector.dimensions = 4


def test_embedding_vector_forbids_extra_fields():
    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        EmbeddingVector(
            values=[0.1],
            dimensions=1,
            unexpected="value",
        )


def test_embedding_batch_serializes_contract():
    batch = EmbeddingBatch.model_validate(
        make_batch_data()
    )

    result = batch.to_dict()

    assert result["provider"] == "ollama"
    assert result["model"] == (
        "qwen3-embedding:0.6b"
    )
    assert result["dimensions"] == 3
    assert result["input_count"] == 2
    assert result["elapsed_ms"] == 12.5
    assert len(result["vectors"]) == 2


def test_embedding_batch_vector_count_must_match():
    data = make_batch_data()
    data["input_count"] = 3

    with pytest.raises(
        ValidationError,
        match="vector count must equal input_count",
    ):
        EmbeddingBatch.model_validate(data)


def test_embedding_batch_vector_dimensions_must_match():
    data = make_batch_data()
    data["dimensions"] = 4

    with pytest.raises(
        ValidationError,
        match=(
            "all vector dimensions must equal "
            "batch dimensions"
        ),
    ):
        EmbeddingBatch.model_validate(data)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("provider", "provider cannot be empty"),
        ("model", "model cannot be empty"),
        (
            "embedding_version",
            "embedding_version cannot be empty",
        ),
    ],
)
def test_embedding_batch_strings_cannot_be_empty(
    field_name: str,
    message: str,
):
    data = make_batch_data()
    data[field_name] = " "

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        EmbeddingBatch.model_validate(data)


def test_embedding_batch_input_count_must_be_positive():
    data = make_batch_data()
    data["input_count"] = 0

    with pytest.raises(
        ValidationError,
        match="input_count must be at least 1",
    ):
        EmbeddingBatch.model_validate(data)


@pytest.mark.parametrize(
    "invalid_elapsed_ms",
    [
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_embedding_batch_elapsed_ms_must_be_valid(
    invalid_elapsed_ms: float,
):
    data = make_batch_data()
    data["elapsed_ms"] = invalid_elapsed_ms

    with pytest.raises(
        ValidationError,
        match=(
            "elapsed_ms must be finite and "
            "non-negative"
        ),
    ):
        EmbeddingBatch.model_validate(data)


def test_embedding_batch_is_frozen():
    batch = EmbeddingBatch.model_validate(
        make_batch_data()
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        batch.input_count = 3


def test_embedding_batch_forbids_extra_fields():
    data = make_batch_data()
    data["unexpected"] = "value"

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        EmbeddingBatch.model_validate(data)

def make_run_report_data() -> dict[str, object]:
    return {
        "provider": "ollama",
        "model": "qwen3-embedding:0.6b",
        "dimensions": 1024,
        "embedding_version": (
            "ollama:qwen3-embedding:0.6b:"
            "dimensions-1024:query-v1-test"
        ),
        "chunk_count": 8,
        "batch_count": 2,
        "vector_count": 8,
        "elapsed_ms": 1200.0,
        "minimum_vector_norm": 0.999999,
        "maximum_vector_norm": 1.000001,
    }


def test_embedding_run_report_serializes():
    report = EmbeddingRunReport.model_validate(
        make_run_report_data()
    )

    result = report.to_dict()

    assert result["chunk_count"] == 8
    assert result["batch_count"] == 2
    assert result["vector_count"] == 8
    assert result["dimensions"] == 1024


def test_run_report_vector_count_must_match():
    data = make_run_report_data()
    data["vector_count"] = 7

    with pytest.raises(
        ValidationError,
        match="vector_count must equal chunk_count",
    ):
        EmbeddingRunReport.model_validate(data)


def test_run_report_batch_count_cannot_exceed_chunks():
    data = make_run_report_data()
    data["batch_count"] = 9

    with pytest.raises(
        ValidationError,
        match=(
            "batch_count cannot exceed chunk_count"
        ),
    ):
        EmbeddingRunReport.model_validate(data)


def test_run_report_norm_range_must_be_ordered():
    data = make_run_report_data()
    data["minimum_vector_norm"] = 1.1
    data["maximum_vector_norm"] = 1.0

    with pytest.raises(
        ValidationError,
        match=(
            "minimum_vector_norm cannot exceed "
            "maximum_vector_norm"
        ),
    ):
        EmbeddingRunReport.model_validate(data)
