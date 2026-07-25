import pytest

from knowledge_normalizer.models import (
    BlockType,
    NormalizedBlock,
)


def build_block(**overrides):
    values = {
        "document_id": "sha256:test",
        "text": "1.1 Test",
        "block_type": (
            BlockType.SECTION_HEADING
        ),
        "heading_path": (
            "Chapter",
            "1.1 Test",
        ),
        "page_start": 1,
        "page_end": 1,
        "ordinal": 1,
        "source_path": "D:/source.pdf",
        "normalization_version": "1.0.0",
    }
    values.update(overrides)
    return NormalizedBlock(**values)


def test_normalized_block_serializes_contract():
    block = build_block()

    assert block.to_dict() == {
        "document_id": "sha256:test",
        "text": "1.1 Test",
        "block_type": "section_heading",
        "heading_path": [
            "Chapter",
            "1.1 Test",
        ],
        "page_start": 1,
        "page_end": 1,
        "ordinal": 1,
        "source_path": "D:/source.pdf",
        "normalization_version": "1.0.0",
    }


def test_block_rejects_invalid_page_range():
    with pytest.raises(
        ValueError,
        match="page_end",
    ):
        build_block(
            page_start=2,
            page_end=1,
        )


def test_heading_must_end_its_path():
    with pytest.raises(
        ValueError,
        match="heading_path",
    ):
        build_block(
            heading_path=(
                "Chapter",
                "Different",
            )
        )

