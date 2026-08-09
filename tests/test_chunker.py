import hashlib

import pytest

from rag_lab.chunking import (
    ChunkingConfig,
    chunk_normalized_blocks,
)
from rag_lab.chunking.chunker import (
    _ATOMIC_BODY_TYPES,
    _BODY_BLOCK_TYPES,
    _CONTROL_BLOCK_TYPES,
    _NON_INDEXABLE_BLOCK_TYPES,
    _SPLITTABLE_BODY_TYPES,
    _CandidateGroup,
    _ChunkDraft,
    _ContentUnit,
    _complete_sentence_suffix,
    _draft_overlap_char_count,
    _draft_to_knowledge_chunk,
    _ends_complete_sentence,
    _group_body_blocks,
    _is_body_block,
    _is_control_block,
    _is_cross_page_paragraph_continuation,
    _join_continuation_text,
    _merge_cross_page_paragraphs,
    _pack_content_units,
    _prepare_chunk_drafts,
    _prepare_oversized_units,
    _render_draft_content,
    _render_draft_index_text,
    _select_overlap_units,
    _sentence_boundary_positions,
    _split_text_to_budget,
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
        | _NON_INDEXABLE_BLOCK_TYPES
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
            max_chars=100,
            overlap_chars=0,
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
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert len(drafts) == 2
    assert drafts[0].units == (units[0],)
    assert drafts[1].units == (units[1],)


def test_whole_unit_overlap_is_added_to_next_draft():
    units = (
        build_unit("A" * 60, 1),
        build_unit(("乙" * 25) + "。", 2),
        build_unit("C" * 60, 3),
    )

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=units,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert len(drafts) == 2
    assert drafts[0].units == units[:2]
    assert drafts[1].units == units[1:]
    assert drafts[1].overlap_unit_count == 1
    assert _draft_overlap_char_count(
        drafts[1]
    ) == 26
    assert all(
        len(_render_draft_index_text(draft))
        <= 100
        for draft in drafts
    )


def test_overlap_uses_complete_sentence_suffix():
    previous = build_unit(
        ("甲" * 40)
        + "。"
        + ("乙" * 20)
        + "。",
        1,
    )
    current = build_unit("C" * 60, 2)

    overlap = _select_overlap_units(
        heading_path=("章节",),
        source_units=(previous,),
        next_unit=current,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert len(overlap) == 1
    assert overlap[0].text == (
        ("乙" * 20) + "。"
    )
    assert overlap[0].blocks == (
        previous.blocks
    )


def test_incomplete_sentence_is_not_cut_for_overlap():
    previous = build_unit("A" * 60, 1)
    current = build_unit("B" * 60, 2)

    overlap = _select_overlap_units(
        heading_path=("章节",),
        source_units=(previous,),
        next_unit=current,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert overlap == ()


def test_atomic_unit_is_not_split_for_overlap():
    block = build_block(
        ordinal=1,
        text="T" * 40,
        block_type=BlockType.TABLE.value,
    )
    atomic = _ContentUnit(
        text=block.text,
        blocks=(block,),
    )
    current = build_unit("B" * 60, 2)

    overlap = _select_overlap_units(
        heading_path=("章节",),
        source_units=(atomic,),
        next_unit=current,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert overlap == ()


def test_overlap_does_not_propagate_to_third_draft():
    units = tuple(
        build_unit(
            ("甲" * 39)
            + "。"
            + (character * 20)
            + "。",
            ordinal,
        )
        for ordinal, character in enumerate(
            ("A", "B", "C"),
            start=1,
        )
    )

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=units,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=25,
        ),
    )

    assert len(drafts) == 3
    assert drafts[1].overlap_unit_count == 1
    assert drafts[2].overlap_unit_count == 1

    third_block_ids = {
        block.block_id
        for unit in drafts[2].units
        for block in unit.blocks
    }

    assert units[0].blocks[0].block_id not in (
        third_block_ids
    )
    assert units[1].blocks[0].block_id in (
        third_block_ids
    )
    assert units[2].blocks[0].block_id in (
        third_block_ids
    )


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
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert len(drafts) == 1


def test_single_oversized_unit_is_preserved():
    unit = build_unit("A" * 120, 1)

    drafts = _pack_content_units(
        heading_path=("章节",),
        units=(unit,),
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
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
            max_chars=100,
            overlap_chars=0,
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
            max_chars=100,
            overlap_chars=0,
        ),
    )

    packed_ordinals = [
        unit.blocks[0].ordinal
        for draft in drafts
        for unit in draft.units
    ]

    assert packed_ordinals == [1, 2, 3]


def test_atomic_and_splittable_types_cover_body_types():
    assert _ATOMIC_BODY_TYPES.isdisjoint(
        _SPLITTABLE_BODY_TYPES
    )

    assert (
        _ATOMIC_BODY_TYPES
        | _SPLITTABLE_BODY_TYPES
    ) == _BODY_BLOCK_TYPES


def test_long_paragraph_prefers_sentence_boundary():
    text = (
        ("甲" * 60)
        + "。"
        + ("乙" * 60)
        + "。"
    )
    unit = build_unit(text, 1)

    result = _prepare_oversized_units(
        heading_path=("章节",),
        units=(unit,),
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert result.long_block_split_count == 1
    assert result.oversized_atomic_block_count == 0
    assert len(result.units) == 2
    assert result.units[0].text == (
        ("甲" * 60) + "。"
    )
    assert result.units[1].text == (
        ("乙" * 60) + "。"
    )

    assert all(
        prepared.blocks == unit.blocks
        for prepared in result.units
    )


def test_long_text_without_sentence_uses_hard_split():
    text = "A" * 200

    fragments = _split_text_to_budget(
        text,
        max_chars=96,
    )

    assert [
        len(fragment)
        for fragment in fragments
    ] == [96, 96, 8]

    assert "".join(fragments) == text


def test_list_item_can_be_split():
    block = build_block(
        ordinal=1,
        text="A" * 120,
        block_type=BlockType.LIST_ITEM.value,
    )
    unit = _ContentUnit(
        text=block.text,
        blocks=(block,),
    )

    result = _prepare_oversized_units(
        heading_path=("章节",),
        units=(unit,),
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert result.long_block_split_count == 1
    assert len(result.units) == 2


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.TABLE.value,
        BlockType.CODE.value,
        BlockType.EQUATION.value,
    ],
)
def test_oversized_atomic_unit_is_preserved(
    block_type,
):
    block = build_block(
        ordinal=1,
        text="A" * 120,
        block_type=block_type,
    )
    unit = _ContentUnit(
        text=block.text,
        blocks=(block,),
    )

    result = _prepare_oversized_units(
        heading_path=("章节",),
        units=(unit,),
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert result.units == (unit,)
    assert result.long_block_split_count == 0
    assert (
        result.oversized_atomic_block_count
        == 1
    )


def test_heading_must_leave_content_capacity():
    with pytest.raises(
        ValueError,
        match="leave room for content",
    ):
        _split_text_to_budget(
            "正文",
            max_chars=0,
        )


def test_decimal_period_is_not_sentence_boundary():
    text = "RTT 为 3.14 秒。下一句。"

    boundaries = _sentence_boundary_positions(
        text
    )

    assert boundaries == (
        text.index("。") + 1,
        len(text),
    )


def test_draft_pipeline_aggregates_processing():
    first = build_block(
        ordinal=1,
        text="A" * 60,
        page_start=19,
        page_end=19,
    )
    second = build_block(
        ordinal=2,
        text=("B" * 60) + "。",
        page_start=20,
        page_end=20,
    )
    atomic = build_block(
        ordinal=3,
        text="T" * 120,
        block_type=BlockType.TABLE.value,
        heading_path=["第二节"],
        page_start=21,
        page_end=21,
    )

    # 故意使用乱序输入，验证完整流水线会先排序。
    result = _prepare_chunk_drafts(
        blocks=[
            atomic,
            second,
            first,
        ],
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert result.cross_page_join_count == 1
    assert result.long_block_split_count == 1
    assert (
        result.oversized_atomic_block_count
        == 1
    )

    assert len(result.drafts) == 3

    assert result.drafts[0].heading_path == (
        "第1章 计算机网络和因特网",
    )
    assert result.drafts[1].heading_path == (
        "第1章 计算机网络和因特网",
    )
    assert result.drafts[2].heading_path == (
        "第二节",
    )

    assert [
        block.ordinal
        for unit in result.drafts[2].units
        for block in unit.blocks
    ] == [3]


def test_control_only_pipeline_produces_no_drafts():
    title = build_block(
        ordinal=1,
        text="计算机网络",
        block_type=(
            BlockType.DOCUMENT_TITLE.value
        ),
        heading_path=["计算机网络"],
    )
    heading = build_block(
        ordinal=2,
        text="第1章",
        block_type=(
            BlockType.SECTION_HEADING.value
        ),
        heading_path=[
            "计算机网络",
            "第1章",
        ],
    )

    result = _prepare_chunk_drafts(
        blocks=[title, heading],
        config=ChunkingConfig(),
    )

    assert result.drafts == ()
    assert result.cross_page_join_count == 0
    assert result.long_block_split_count == 0
    assert (
        result.oversized_atomic_block_count
        == 0
    )


def test_normal_pipeline_respects_max_chars():
    first = build_block(
        ordinal=1,
        text="A" * 40,
    )
    second = build_block(
        ordinal=2,
        text="B" * 40,
    )

    result = _prepare_chunk_drafts(
        blocks=[first, second],
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert len(result.drafts) == 1
    assert result.cross_page_join_count == 0
    assert result.long_block_split_count == 0
    assert (
        result.oversized_atomic_block_count
        == 0
    )

    assert len(
        _render_draft_index_text(
            result.drafts[0]
        )
    ) <= 100

def test_draft_converts_to_knowledge_chunk():
    first = build_block(
        ordinal=1,
        text="第一段",
        page_start=19,
        page_end=19,
    )
    second = build_block(
        ordinal=2,
        text="第二段",
        page_start=20,
        page_end=20,
    )

    draft = _ChunkDraft(
        heading_path=tuple(
            first.heading_path
        ),
        units=(
            _ContentUnit(
                text=first.text,
                blocks=(first,),
            ),
            _ContentUnit(
                text=second.text,
                blocks=(second,),
            ),
        ),
    )

    chunk = _draft_to_knowledge_chunk(
        draft=draft,
        ordinal=1,
        config=ChunkingConfig(),
    )

    expected_content = "第一段\n\n第二段"
    expected_hash = (
        "sha256:"
        + hashlib.sha256(
            expected_content.encode("utf-8")
        ).hexdigest()
    )

    assert chunk.content == expected_content
    assert chunk.index_text.endswith(
        expected_content
    )
    assert chunk.page_start == 19
    assert chunk.page_end == 20
    assert chunk.ordinal == 1
    assert chunk.block_ids == [
        first.block_id,
        second.block_id,
    ]
    assert chunk.content_hash == expected_hash


def test_chunk_id_does_not_depend_on_ordinal():
    unit = build_unit("正文", 1)
    draft = _ChunkDraft(
        heading_path=("章节",),
        units=(unit,),
    )

    first = _draft_to_knowledge_chunk(
        draft=draft,
        ordinal=1,
        config=ChunkingConfig(),
    )
    later = _draft_to_knowledge_chunk(
        draft=draft,
        ordinal=99,
        config=ChunkingConfig(),
    )

    assert first.chunk_id == later.chunk_id
    assert first.ordinal == 1
    assert later.ordinal == 99


def test_content_change_changes_chunk_id():
    block = build_block(
        ordinal=1,
        text="原始正文",
    )

    original = _ChunkDraft(
        heading_path=("章节",),
        units=(
            _ContentUnit(
                text="原始正文",
                blocks=(block,),
            ),
        ),
    )
    changed = _ChunkDraft(
        heading_path=("章节",),
        units=(
            _ContentUnit(
                text="修改后的正文",
                blocks=(block,),
            ),
        ),
    )

    original_chunk = _draft_to_knowledge_chunk(
        draft=original,
        ordinal=1,
        config=ChunkingConfig(),
    )
    changed_chunk = _draft_to_knowledge_chunk(
        draft=changed,
        ordinal=1,
        config=ChunkingConfig(),
    )

    assert (
        original_chunk.content_hash
        != changed_chunk.content_hash
    )
    assert (
        original_chunk.chunk_id
        != changed_chunk.chunk_id
    )


def test_public_chunker_returns_result_and_report():
    blocks = [
        build_block(
            ordinal=1,
            text="A" * 40,
        ),
        build_block(
            ordinal=2,
            text="B" * 40,
        ),
    ]

    result = chunk_normalized_blocks(
        blocks=blocks,
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].ordinal == 1
    assert result.report.document_id == (
        "sha256:document"
    )
    assert result.report.input_block_count == 2
    assert result.report.output_chunk_count == 1


def test_public_chunker_reports_overlap_provenance():
    first = build_block(
        ordinal=1,
        text=(
            ("甲" * 40)
            + "。"
            + ("乙" * 20)
            + "。"
        ),
    )
    second = build_block(
        ordinal=2,
        text="B" * 60,
    )

    result = chunk_normalized_blocks(
        blocks=[first, second],
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert len(result.chunks) == 2
    assert result.chunks[1].content.startswith(
        ("乙" * 20) + "。"
    )
    assert result.chunks[1].block_ids == [
        first.block_id,
        second.block_id,
    ]
    assert result.report.overlapped_chunk_count == 1
    assert result.report.overlap_char_count == 21


def test_overlap_does_not_cross_heading_path():
    first = build_block(
        ordinal=1,
        text=("甲" * 20) + "。",
        heading_path=["第一节"],
    )
    second = build_block(
        ordinal=2,
        text="B" * 90,
        heading_path=["第二节"],
    )

    result = chunk_normalized_blocks(
        blocks=[first, second],
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=30,
        ),
    )

    assert len(result.chunks) == 2
    assert result.chunks[1].block_ids == [
        second.block_id,
    ]
    assert result.report.overlapped_chunk_count == 0
    assert result.report.overlap_char_count == 0


def test_zero_overlap_preserves_nonoverlapping_output():
    first = build_block(
        ordinal=1,
        text=("甲" * 20) + "。",
        heading_path=["章节"],
    )
    second = build_block(
        ordinal=2,
        text="B" * 90,
        heading_path=["章节"],
    )

    result = chunk_normalized_blocks(
        blocks=[first, second],
        config=ChunkingConfig(
            max_chars=100,
            overlap_chars=0,
        ),
    )

    assert [
        chunk.block_ids
        for chunk in result.chunks
    ] == [
        [first.block_id],
        [second.block_id],
    ]
    assert result.report.overlapped_chunk_count == 0
    assert result.report.overlap_char_count == 0


def test_inserting_earlier_group_keeps_later_id():
    earlier = build_block(
        ordinal=1,
        text="较早正文",
        heading_path=["第一节"],
    )
    later = build_block(
        ordinal=2,
        text="较晚正文",
        heading_path=["第二节"],
    )

    later_only = chunk_normalized_blocks(
        blocks=[later]
    )
    complete = chunk_normalized_blocks(
        blocks=[earlier, later]
    )

    assert (
        later_only.chunks[0].chunk_id
        == complete.chunks[1].chunk_id
    )
    assert later_only.chunks[0].ordinal == 1
    assert complete.chunks[1].ordinal == 2


def test_control_only_document_returns_empty_chunks():
    title = build_block(
        ordinal=1,
        text="计算机网络",
        block_type=(
            BlockType.DOCUMENT_TITLE.value
        ),
        heading_path=["计算机网络"],
    )

    result = chunk_normalized_blocks(
        blocks=[title]
    )

    assert result.chunks == []
    assert result.report.input_block_count == 1
    assert result.report.output_chunk_count == 0


def test_figure_labels_are_preserved_but_not_chunked():
    label = build_block(
        ordinal=1,
        text="La",
        block_type=BlockType.FIGURE_LABEL.value,
    )

    result = chunk_normalized_blocks(blocks=[label])

    assert result.chunks == []
    assert result.report.input_block_count == 1
    assert result.report.output_chunk_count == 0
