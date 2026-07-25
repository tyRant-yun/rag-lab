import pytest

from rag_lab.chunking import ChunkingConfig
from rag_lab.chunking.chunker import (
    _BODY_BLOCK_TYPES,
    _CONTROL_BLOCK_TYPES,
    _CandidateGroup,
    _ChunkDraft,
    _ContentUnit,
    _ends_complete_sentence,
    _group_body_blocks,
    _is_body_block,
    _is_control_block,
    _is_cross_page_paragraph_continuation,
    _join_continuation_text,
    _merge_cross_page_paragraphs,
    _pack_content_units,
    _render_draft_content,
    _render_draft_index_text,
    _validate_and_sort_blocks,
)
from rag_lab.contracts import (
    BlockType,
    NormalizedBlock,
)


def build_group(
    *blocks: NormalizedBlock,
) -> _CandidateGroup:
    return _CandidateGroup(
        heading_path=tuple(
            blocks[0].heading_path
        ),
        blocks=blocks,
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

def build_unit(
    text: str,
    ordinal: int,
) -> _ContentUnit:
    block = build_block(
        ordinal=ordinal,
        text=text,
    )

    return _ContentUnit(
        text=text,
        blocks=(block,),
    )


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


def test_same_heading_body_blocks_share_group():
    blocks = [
        build_block(ordinal=1),
        build_block(
            ordinal=2,
            block_type=BlockType.LIST_ITEM.value,
        ),
    ]

    groups = _group_body_blocks(blocks)

    assert len(groups) == 1
    assert groups[0].heading_path == (
        "第1章 计算机网络和因特网",
    )
    assert [
        block.ordinal
        for block in groups[0].blocks
    ] == [1, 2]


def test_heading_path_change_starts_new_group():
    blocks = [
        build_block(
            ordinal=1,
            heading_path=["第1章", "1.1"],
        ),
        build_block(
            ordinal=2,
            heading_path=["第1章", "1.2"],
        ),
    ]

    groups = _group_body_blocks(blocks)

    assert len(groups) == 2
    assert groups[0].heading_path == (
        "第1章",
        "1.1",
    )
    assert groups[1].heading_path == (
        "第1章",
        "1.2",
    )


def test_control_block_ends_current_group():
    blocks = [
        build_block(
            ordinal=1,
            heading_path=["第1章", "1.1"],
        ),
        build_block(
            ordinal=2,
            text="1.1",
            block_type=(
                BlockType.SECTION_HEADING.value
            ),
            heading_path=["第1章", "1.1"],
        ),
        build_block(
            ordinal=3,
            heading_path=["第1章", "1.1"],
        ),
    ]

    groups = _group_body_blocks(blocks)

    assert len(groups) == 2
    assert [
        block.ordinal
        for block in groups[0].blocks
    ] == [1]
    assert [
        block.ordinal
        for block in groups[1].blocks
    ] == [3]


def test_control_only_input_produces_no_groups():
    blocks = [
        build_block(
            ordinal=1,
            text="第1章",
            block_type=(
                BlockType.DOCUMENT_TITLE.value
            ),
            heading_path=["第1章"],
        ),
        build_block(
            ordinal=2,
            text="1.1",
            block_type=(
                BlockType.SECTION_HEADING.value
            ),
            heading_path=["第1章", "1.1"],
        ),
    ]

    assert _group_body_blocks(blocks) == []


def test_grouping_sorts_blocks_first():
    third = build_block(ordinal=3)
    first = build_block(ordinal=1)
    second = build_block(ordinal=2)

    groups = _group_body_blocks(
        [third, first, second]
    )

    assert [
        block.ordinal
        for block in groups[0].blocks
    ] == [1, 2, 3]


def test_cross_page_chinese_paragraph_is_joined():
    previous = build_block(
        ordinal=1,
        text="这些媒体用于存",
        page_start=19,
        page_end=19,
    )
    current = build_block(
        ordinal=2,
        text="储和传输数据。",
        page_start=20,
        page_end=20,
    )

    units, join_count = (
        _merge_cross_page_paragraphs(
            build_group(previous, current)
        )
    )

    assert join_count == 1
    assert len(units) == 1
    assert units[0].text == (
        "这些媒体用于存储和传输数据。"
    )
    assert units[0].blocks == (
        previous,
        current,
    )


def test_complete_sentence_is_not_joined():
    previous = build_block(
        ordinal=1,
        text="这是完整句子。”",
        page_start=19,
        page_end=19,
    )
    current = build_block(
        ordinal=2,
        text="这是新的段落。",
        page_start=20,
        page_end=20,
    )

    assert _ends_complete_sentence(
        previous.text
    )

    assert not (
        _is_cross_page_paragraph_continuation(
            previous,
            current,
        )
    )


def test_same_page_paragraphs_are_not_joined():
    previous = build_block(
        ordinal=1,
        text="第一个段落",
        page_start=19,
        page_end=19,
    )
    current = build_block(
        ordinal=2,
        text="第二个段落",
        page_start=19,
        page_end=19,
    )

    units, join_count = (
        _merge_cross_page_paragraphs(
            build_group(previous, current)
        )
    )

    assert join_count == 0
    assert len(units) == 2


def test_nonconsecutive_ordinals_are_not_joined():
    previous = build_block(
        ordinal=1,
        text="上一页内容",
        page_start=19,
        page_end=19,
    )
    current = build_block(
        ordinal=3,
        text="下一页内容",
        page_start=20,
        page_end=20,
    )

    assert not (
        _is_cross_page_paragraph_continuation(
            previous,
            current,
        )
    )


def test_nonparagraph_blocks_are_not_joined():
    previous = build_block(
        ordinal=1,
        text="列表内容",
        block_type=BlockType.LIST_ITEM.value,
        page_start=19,
        page_end=19,
    )
    current = build_block(
        ordinal=2,
        text="下一页内容",
        page_start=20,
        page_end=20,
    )

    assert not (
        _is_cross_page_paragraph_continuation(
            previous,
            current,
        )
    )


def test_ascii_boundaries_receive_space():
    assert _join_continuation_text(
        "the",
        "network",
    ) == "the network"


def test_three_page_continuation_forms_one_unit():
    first = build_block(
        ordinal=1,
        text="用于",
        page_start=19,
        page_end=19,
    )
    second = build_block(
        ordinal=2,
        text="存储和",
        page_start=20,
        page_end=20,
    )
    third = build_block(
        ordinal=3,
        text="传输数据。",
        page_start=21,
        page_end=21,
    )

    units, join_count = (
        _merge_cross_page_paragraphs(
            build_group(
                first,
                second,
                third,
            )
        )
    )

    assert join_count == 2
    assert len(units) == 1
    assert units[0].text == (
        "用于存储和传输数据。"
    )
    assert units[0].blocks == (
        first,
        second,
        third,
    )


def test_draft_rendering_includes_heading_context():
    draft = _ChunkDraft(
        heading_path=(
            "第1章",
            "1.1 因特网",
        ),
        units=(
            build_unit("第一段", 1),
            build_unit("第二段", 2),
        ),
    )

    assert _render_draft_content(draft) == (
        "第一段\n\n第二段"
    )
    assert _render_draft_index_text(draft) == (
        "第1章\n"
        "1.1 因特网\n\n"
        "第一段\n\n"
        "第二段"
    )


def test_units_that_fit_share_one_draft():
    units = (
        build_unit("A" * 40, 1),
        build_unit("B" * 40, 2),
    )

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=units,
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    assert len(drafts) == 1
    assert drafts[0].units == units
    assert len(
        _render_draft_index_text(drafts[0])
    ) <= 100


def test_overflow_starts_new_draft():
    units = (
        build_unit("A" * 60, 1),
        build_unit("B" * 60, 2),
    )

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=units,
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    assert len(drafts) == 2
    assert drafts[0].units == (units[0],)
    assert drafts[1].units == (units[1],)


def test_exact_limit_is_allowed():
    draft = _ChunkDraft(
        heading_path=("章节",),
        units=(
            build_unit("A" * 96, 1),
        ),
    )

    assert len(
        _render_draft_index_text(draft)
    ) == 100

    drafts = _pack_content_units(
        heading_path=draft.heading_path,
        units=draft.units,
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    assert len(drafts) == 1


def test_single_oversized_unit_is_preserved():
    unit = build_unit("A" * 120, 1)

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=(unit,),
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    assert len(drafts) == 1
    assert drafts[0].units == (unit,)
    assert len(
        _render_draft_index_text(drafts[0])
    ) > 100


def test_empty_units_produce_no_drafts():
    drafts = _pack_content_units(
        heading_path=("章节",),
        units=(),
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    assert drafts == ()


def test_packing_preserves_unit_order():
    units = (
        build_unit("A" * 60, 1),
        build_unit("B" * 60, 2),
        build_unit("C" * 20, 3),
    )

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=units,
        config=ChunkingConfig(
            max_chars=100
        ),
    )

    packed_ordinals = [
        unit.blocks[0].ordinal
        for draft in drafts
        for unit in draft.units
    ]

    assert packed_ordinals == [1, 2, 3]
