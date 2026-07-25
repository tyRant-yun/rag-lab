import json
from pathlib import Path

from knowledge_normalizer.models import (
    BlockType,
)
from knowledge_normalizer.normalizer import (
    normalize_docling_document,
    normalize_text,
)
from knowledge_normalizer.serialization import (
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


def test_cross_page_paragraphs_are_merged(
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

    merged = next(
        block
        for block in result.blocks
        if block.text
        == "第一段未结束继续到下一页。"
    )

    assert merged.page_start == 19
    assert merged.page_end == 20
    assert (
        result.report.merged_cross_page_count
        == 1
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
