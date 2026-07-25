import json
from copy import deepcopy
from pathlib import Path

from rag_lab.contracts.blocks import (
    BlockType,
)
from rag_lab.normalization.normalizer import (
    normalize_docling_document,
    normalize_text,
)
from rag_lab.normalization.serialization import (
    write_normalization_outputs,
)


def provenance(
    *,
    page: int,
    top: float,
    bottom: float,
) -> list[dict[str, object]]:
    return [
        {
            "page_no": page,
            "bbox": {
                "l": 50.0,
                "t": top,
                "r": 450.0,
                "b": bottom,
                "coord_origin": "BOTTOMLEFT",
            },
            "charspan": [0, 10],
        }
    ]


def text_item(
    index: int,
    *,
    text: str,
    page: int,
    top: float,
    label: str = "text",
    content_layer: str = "body",
    parent_ref: str = "#/body",
) -> dict[str, object]:
    return {
        "self_ref": f"#/texts/{index}",
        "parent": {"$ref": parent_ref},
        "children": [],
        "content_layer": content_layer,
        "label": label,
        "prov": provenance(
            page=page,
            top=top,
            bottom=top - 30,
        ),
        "orig": text,
        "text": text,
    }


def sample_document() -> dict[str, object]:
    return {
        "pages": {
            "19": {
                "page_no": 19,
                "size": {
                    "width": 500,
                    "height": 700,
                },
            },
            "20": {
                "page_no": 20,
                "size": {
                    "width": 500,
                    "height": 700,
                },
            },
        },
        "texts": [
            text_item(
                0,
                text="第 1 章",
                page=19,
                top=680,
                label="page_header",
                content_layer="furniture",
            ),
            text_item(
                1,
                text="计算机网络和因特网",
                page=19,
                top=640,
                label="section_header",
            ),
            # Intentionally store the lower paragraph first.
            text_item(
                2,
                text="第二段。",
                page=19,
                top=420,
            ),
            text_item(
                3,
                text="1. 1 什么是因特网",
                page=19,
                top=600,
                label="section_header",
            ),
            text_item(
                4,
                text="第一段未结束",
                page=19,
                top=520,
            ),
            text_item(
                5,
                text="继续到下一页。",
                page=20,
                top=650,
            ),
            text_item(
                6,
                text="1. 1. 1 具体构成描述",
                page=20,
                top=580,
                label="section_header",
            ),
            text_item(
                7,
                text="用 户使用 TCP / IP 协议 。",
                page=20,
                top=500,
            ),
            text_item(
                8,
                text="1. 人类活动的类比",
                page=20,
                top=430,
                label="section_header",
            ),
            text_item(
                9,
                text="图例：",
                page=20,
                top=200,
                label="section_header",
                parent_ref="#/pictures/0",
            ),
            text_item(
                10,
                text="图 1-1 示例",
                page=20,
                top=100,
                label="caption",
                parent_ref="#/pictures/0",
            ),
        ],
    }


def test_normalize_text_handles_cjk_spacing():
    assert normalize_text(
        "用 户使用 TCP / IP 协议 。"
    ) == "用户使用 TCP/IP 协议。"

    assert normalize_text(
        "系统 ， 该系统 （ protocol ） 工作。"
    ) == "系统，该系统（protocol）工作。"

    assert normalize_text(
        "1. 1. 1 具体构成描述"
    ) == "1.1.1 具体构成描述"


def test_normalizer_restores_order_and_contract(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")

    result = normalize_docling_document(
        docling_document=sample_document(),
        source_path=source,
        normalization_version="1.0.0",
    )

    assert [
        block.ordinal
        for block in result.blocks
    ] == list(
        range(1, len(result.blocks) + 1)
    )

    assert result.blocks[0].block_type == (
        BlockType.DOCUMENT_TITLE.value
    )
    assert result.blocks[0].text == (
        "第1章 计算机网络和因特网"
    )

    texts = [
        block.text
        for block in result.blocks
    ]

    assert texts.index("第一段未结束") < (
        texts.index("第二段。")
    )
    assert "用户使用 TCP/IP 协议。" in texts
    assert "图例：" not in texts

    false_heading = next(
        block
        for block in result.blocks
        if block.text
        == "1. 人类活动的类比"
    )
    assert false_heading.block_type == (
        BlockType.PARAGRAPH.value
    )
    assert false_heading.heading_path[-1] == (
        "1.1.1 具体构成描述"
    )

    heading = next(
        block
        for block in result.blocks
        if block.text
        == "1.1.1 具体构成描述"
    )
    assert heading.heading_path == [
        "第1章 计算机网络和因特网",
        "1.1 什么是因特网",
        "1.1.1 具体构成描述",
    ]

    assert all(
        block.block_id.startswith("sha256:")
        for block in result.blocks
    )
    assert len(
        {
            block.block_id
            for block in result.blocks
        }
    ) == len(result.blocks)
    assert all(
        block.image_path is None
        for block in result.blocks
    )

    assert result.report.reordered_block_count > 0
    assert result.report.removed_furniture_count == 1
    assert result.report.downgraded_heading_count == 1
    assert result.report.pages_requiring_review == (
        20,
    )


def test_cross_page_paragraphs_remain_separate(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")
    document = sample_document()
    texts = document["texts"]
    assert isinstance(texts, list)

    # Remove the intervening lower paragraph so the unfinished
    # paragraph is adjacent to the next page paragraph.
    texts.pop(2)

    result = normalize_docling_document(
        docling_document=document,
        source_path=source,
        normalization_version="1.0.0",
    )

    first = next(
        block
        for block in result.blocks
        if block.text == "第一段未结束"
    )
    continuation = next(
        block
        for block in result.blocks
        if block.text == "继续到下一页。"
    )

    assert first.page_start == 19
    assert first.page_end == 19
    assert continuation.page_start == 20
    assert continuation.page_end == 20
    assert continuation.ordinal == first.ordinal + 1


def test_inserting_earlier_block_keeps_later_ids(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")
    original = sample_document()
    modified = deepcopy(original)
    texts = modified["texts"]
    assert isinstance(texts, list)
    texts.append(
        text_item(
            11,
            text="新增的前置段落。",
            page=19,
            top=550,
        )
    )

    before = normalize_docling_document(
        docling_document=original,
        source_path=source,
        normalization_version="1.0.0",
    )
    after = normalize_docling_document(
        docling_document=modified,
        source_path=source,
        normalization_version="1.0.0",
    )
    before_by_text = {
        block.text: block
        for block in before.blocks
    }
    after_by_text = {
        block.text: block
        for block in after.blocks
    }

    for text in (
        "第一段未结束",
        "第二段。",
        "1.1.1 具体构成描述",
        "用户使用 TCP/IP 协议。",
    ):
        assert (
            before_by_text[text].block_id
            == after_by_text[text].block_id
        )

    assert (
        before_by_text["第一段未结束"].ordinal
        != after_by_text["第一段未结束"].ordinal
    )


def test_caption_receives_relative_image_path(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")
    artifact_directory = tmp_path / "artifact"
    assets = artifact_directory / "assets"
    assets.mkdir(parents=True)
    image = assets / "figure.png"
    image.write_bytes(b"png")
    document = sample_document()
    document["pictures"] = [
        {
            "self_ref": "#/pictures/0",
            "captions": [
                {"$ref": "#/texts/10"}
            ],
            "image": {
                "uri": str(image),
                "mimetype": "image/png",
            },
        }
    ]

    result = normalize_docling_document(
        docling_document=document,
        source_path=source,
        normalization_version="1.0.0",
        artifact_directory=artifact_directory,
    )
    caption = next(
        block
        for block in result.blocks
        if block.block_type
        == BlockType.FIGURE_CAPTION.value
    )

    assert caption.image_path == (
        "assets/figure.png"
    )

    output = tmp_path / "normalized"
    write_normalization_outputs(
        result=result,
        output_directory=output,
        asset_source_directory=(
            artifact_directory
        ),
    )
    markdown = (
        output / "document.md"
    ).read_text(encoding="utf-8")
    assert (
        "![图 1-1 示例]"
        "(assets/figure.png)"
        in markdown
    )
    assert (
        output / "assets" / "figure.png"
    ).read_bytes() == b"png"


def test_multiple_chapters_reset_heading_path(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")
    document = {
        "pages": {
            str(page): {
                "page_no": page,
                "size": {
                    "width": 500,
                    "height": 700,
                },
            }
            for page in (1, 2)
        },
        "texts": [
            text_item(
                0,
                text="第 1 章",
                page=1,
                top=680,
                label="page_header",
                content_layer="furniture",
            ),
            text_item(
                1,
                text="第一章标题",
                page=1,
                top=640,
                label="section_header",
            ),
            text_item(
                2,
                text="1. 1 第一节",
                page=1,
                top=600,
                label="section_header",
            ),
            text_item(
                3,
                text="第一章正文。",
                page=1,
                top=540,
            ),
            text_item(
                4,
                text="第 2 章",
                page=2,
                top=680,
                label="page_header",
                content_layer="furniture",
            ),
            text_item(
                5,
                text="第二章标题",
                page=2,
                top=640,
                label="section_header",
            ),
            text_item(
                6,
                text="2. 1 第二节",
                page=2,
                top=600,
                label="section_header",
            ),
            text_item(
                7,
                text="第二章正文。",
                page=2,
                top=540,
            ),
        ],
    }

    result = normalize_docling_document(
        docling_document=document,
        source_path=source,
        normalization_version="1.0.0",
    )
    second_chapter = next(
        block
        for block in result.blocks
        if block.text == "第2章 第二章标题"
    )
    second_section = next(
        block
        for block in result.blocks
        if block.text == "2.1 第二节"
    )
    second_body = next(
        block
        for block in result.blocks
        if block.text == "第二章正文。"
    )

    assert second_chapter.block_type == (
        BlockType.SECTION_HEADING.value
    )
    assert second_chapter.heading_path == [
        "第2章 第二章标题"
    ]
    assert second_section.heading_path == [
        "第2章 第二章标题",
        "2.1 第二节",
    ]
    assert second_body.heading_path == [
        "第2章 第二章标题",
        "2.1 第二节",
    ]
    assert (
        result.report.downgraded_heading_count
        == 0
    )


def test_outputs_are_deterministic(
    tmp_path: Path,
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")

    result = normalize_docling_document(
        docling_document=sample_document(),
        source_path=source,
        normalization_version="1.0.0",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_normalization_outputs(
        result=result,
        output_directory=first,
    )
    write_normalization_outputs(
        result=result,
        output_directory=second,
    )

    for filename in (
        "blocks.jsonl",
        "document.md",
        "normalization-report.json",
    ):
        assert (
            first / filename
        ).read_bytes() == (
            second / filename
        ).read_bytes()

    records = [
        json.loads(line)
        for line in (
            first / "blocks.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert records[0].keys() == {
        "block_id",
        "document_id",
        "text",
        "block_type",
        "heading_path",
        "page_start",
        "page_end",
        "ordinal",
        "source_path",
        "image_path",
        "normalization_version",
    }
