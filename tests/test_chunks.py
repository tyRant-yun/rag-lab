import pytest

from rag_lab.contracts import KnowledgeChunk


def build_chunk(**overrides):
    values = {
        "chunk_id": "sha256:chunk",
        "document_id": "sha256:document",
        "content": "这是一个用于测试的知识片段。",
        "index_text": (
            "第1章 计算机网络和因特网\n"
            "1.1 什么是因特网\n\n"
            "这是一个用于测试的知识片段。"
        ),
        "heading_path": [
            "第1章 计算机网络和因特网",
            "1.1 什么是因特网",
        ],
        "page_start": 19,
        "page_end": 20,
        "ordinal": 1,
        "block_ids": [
            "sha256:block-1",
            "sha256:block-2",
        ],
        "source_path": "D:/source.pdf",
        "content_hash": "sha256:content",
        "normalization_version": "1.0.0",
        "chunking_version": "1.0.0",
    }
    values.update(overrides)
    return KnowledgeChunk(**values)


def test_knowledge_chunk_serializes_contract():
    chunk = build_chunk()

    result = chunk.to_dict()

    assert result["chunk_id"] == "sha256:chunk"
    assert result["page_start"] == 19
    assert result["page_end"] == 20
    assert result["block_ids"] == [
        "sha256:block-1",
        "sha256:block-2",
    ]


def test_chunk_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="content cannot be empty",
    ):
        build_chunk(content="   ")


def test_chunk_rejects_empty_block_ids():
    with pytest.raises(
        ValueError,
        match="block_ids cannot be empty",
    ):
        build_chunk(block_ids=[])


def test_chunk_rejects_duplicate_block_ids():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_chunk(
            block_ids=[
                "sha256:block-1",
                "sha256:block-1",
            ]
        )


def test_chunk_rejects_invalid_page_range():
    with pytest.raises(
        ValueError,
        match="page_end",
    ):
        build_chunk(
            page_start=20,
            page_end=19,
        )
