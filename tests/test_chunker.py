import pytest

from rag_lab.chunking.chunker import (
    _BODY_BLOCK_TYPES,
    _CONTROL_BLOCK_TYPES,
    _is_body_block,
    _is_control_block,
    _validate_and_sort_blocks,
)
from rag_lab.contracts import (
    BlockType,
    NormalizedBlock,
)


def build_block(
    ordinal: int,
    **overrides,
) -> NormalizedBlock:
    values = {
        "block_id": f"sha256:block-{ordinal}",
        "document_id": "sha256:document",
        "text": f"第 {ordinal} 个段落",
        "block_type": BlockType.PARAGRAPH.value,
        "heading_path": [
            "第1章 计算机网络和因特网",
        ],
        "page_start": 19,
        "page_end": 19,
        "ordinal": ordinal,
        "source_path": "D:/source.pdf",
        "image_path": None,
        "normalization_version": "1.0.0",
    }
    values.update(overrides)
    return NormalizedBlock(**values)


def test_blocks_are_sorted_without_mutating_input():
    second = build_block(ordinal=2)
    first = build_block(ordinal=1)
    original = [second, first]

    result = _validate_and_sort_blocks(original)

    assert [
        block.ordinal
        for block in result
    ] == [1, 2]

    assert original == [second, first]
    assert result is not original


def test_empty_input_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        _validate_and_sort_blocks([])


def test_non_normalized_block_is_rejected():
    with pytest.raises(
        TypeError,
        match="NormalizedBlock",
    ):
        _validate_and_sort_blocks([object()])


def test_multiple_documents_are_rejected():
    blocks = [
        build_block(ordinal=1),
        build_block(
            ordinal=2,
            document_id="sha256:other-document",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="one document",
    ):
        _validate_and_sort_blocks(blocks)


def test_multiple_source_paths_are_rejected():
    blocks = [
        build_block(ordinal=1),
        build_block(
            ordinal=2,
            source_path="D:/other.pdf",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="source_path",
    ):
        _validate_and_sort_blocks(blocks)


def test_multiple_normalization_versions_are_rejected():
    blocks = [
        build_block(ordinal=1),
        build_block(
            ordinal=2,
            normalization_version="2.0.0",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="normalization_version",
    ):
        _validate_and_sort_blocks(blocks)


def test_duplicate_block_ids_are_rejected():
    blocks = [
        build_block(
            ordinal=1,
            block_id="sha256:same",
        ),
        build_block(
            ordinal=2,
            block_id="sha256:same",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="block_id",
    ):
        _validate_and_sort_blocks(blocks)


def test_duplicate_ordinals_are_rejected():
    blocks = [
        build_block(
            ordinal=1,
            block_id="sha256:first",
        ),
        build_block(
            ordinal=1,
            block_id="sha256:second",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="ordinal",
    ):
        _validate_and_sort_blocks(blocks)

@pytest.mark.parametrize(
    (
        "block_type",
        "is_control",
    ),
    [
        (
            BlockType.DOCUMENT_TITLE.value,
            True,
        ),
        (
            BlockType.SECTION_HEADING.value,
            True,
        ),
        (
            BlockType.PARAGRAPH.value,
            False,
        ),
        (
            BlockType.LIST_ITEM.value,
            False,
        ),
        (
            BlockType.FIGURE_CAPTION.value,
            False,
        ),
        (
            BlockType.TABLE.value,
            False,
        ),
        (
            BlockType.CODE.value,
            False,
        ),
        (
            BlockType.EQUATION.value,
            False,
        ),
    ],
)
def test_block_role_is_classified(
    block_type,
    is_control,
):
    overrides = {
        "block_type": block_type,
    }

    if is_control:
        overrides.update(
            {
                "text": "测试标题",
                "heading_path": ["测试标题"],
            }
        )

    block = build_block(
        ordinal=1,
        **overrides,
    )

    assert _is_control_block(block) is is_control
    assert _is_body_block(block) is not is_control


def test_block_roles_cover_all_supported_types():
    supported_types = {
        member.value
        for member in BlockType
    }

    assert _CONTROL_BLOCK_TYPES.isdisjoint(
        _BODY_BLOCK_TYPES
    )

    assert (
        _CONTROL_BLOCK_TYPES
        | _BODY_BLOCK_TYPES
    ) == supported_types
