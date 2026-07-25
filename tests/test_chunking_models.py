import pytest

from rag_lab.chunking import (
    ChunkingConfig,
    ChunkingReport,
    ChunkingResult,
)


def build_report(**overrides):
    values = {
        "document_id": "sha256:document",
        "input_block_count": 10,
        "output_chunk_count": 3,
        "cross_page_join_count": 1,
        "long_block_split_count": 0,
        "oversized_atomic_block_count": 0,
        "overlapped_chunk_count": 2,
        "overlap_char_count": 180,
    }
    values.update(overrides)
    return ChunkingReport(**values)


def test_chunking_config_has_stable_defaults():
    config = ChunkingConfig()

    assert config.max_chars == 1200
    assert config.overlap_chars == 120
    assert config.chunking_version == "1.1.0"


def test_chunking_config_rejects_tiny_limit():
    with pytest.raises(
        ValueError,
        match="max_chars",
    ):
        ChunkingConfig(max_chars=99)


def test_chunking_config_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="overlap_chars",
    ):
        ChunkingConfig(overlap_chars=-1)


def test_chunking_config_rejects_overlap_at_limit():
    with pytest.raises(
        ValueError,
        match="less than max_chars",
    ):
        ChunkingConfig(
            max_chars=100,
            overlap_chars=100,
        )


def test_chunking_report_rejects_negative_count():
    with pytest.raises(ValueError):
        build_report(input_block_count=-1)


def test_empty_chunking_result_is_valid():
    report = build_report(
        input_block_count=0,
        output_chunk_count=0,
    )

    result = ChunkingResult(
        chunks=[],
        report=report,
    )

    assert result.chunks == []
    assert result.report.document_id == (
        "sha256:document"
    )
