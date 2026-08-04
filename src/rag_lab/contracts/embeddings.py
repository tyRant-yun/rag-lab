from __future__ import annotations

import math
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


class EmbeddingVector(BaseModel):
    """One validated embedding vector."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    values: list[float]
    dimensions: int

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.dimensions < 1:
            raise ValueError(
                "dimensions must be at least 1"
            )

        if not self.values:
            raise ValueError(
                "values cannot be empty"
            )

        if len(self.values) != self.dimensions:
            raise ValueError(
                "values length must equal dimensions"
            )

        if any(
            not math.isfinite(value)
            for value in self.values
        ):
            raise ValueError(
                "values must contain only finite numbers"
            )

        if all(
            value == 0.0
            for value in self.values
        ):
            raise ValueError(
                "values cannot be an all-zero vector"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class EmbeddingBatch(BaseModel):
    """Validated output from one embedding operation."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    provider: str
    model: str
    dimensions: int
    vectors: list[EmbeddingVector]
    input_count: int
    elapsed_ms: float
    embedding_version: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.provider.strip():
            raise ValueError(
                "provider cannot be empty"
            )

        if not self.model.strip():
            raise ValueError(
                "model cannot be empty"
            )

        if not self.embedding_version.strip():
            raise ValueError(
                "embedding_version cannot be empty"
            )

        if self.dimensions < 1:
            raise ValueError(
                "dimensions must be at least 1"
            )

        if self.input_count < 1:
            raise ValueError(
                "input_count must be at least 1"
            )

        if len(self.vectors) != self.input_count:
            raise ValueError(
                "vector count must equal input_count"
            )

        if any(
            vector.dimensions != self.dimensions
            for vector in self.vectors
        ):
            raise ValueError(
                "all vector dimensions must equal "
                "batch dimensions"
            )

        if (
            not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0
        ):
            raise ValueError(
                "elapsed_ms must be finite and "
                "non-negative"
            )

        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
