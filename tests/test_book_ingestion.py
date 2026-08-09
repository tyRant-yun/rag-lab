from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_lab.ingestion.manifest import (
    BookManifest,
    BookSection,
    read_book_manifest,
)
from rag_lab.ingestion.pipeline import (
    SectionBuildResult,
    assemble_corpus,
    process_book,
    process_existing_docling_book,
)
from rag_lab.knowledge_base import (
    PublicKnowledgeBaseInfo,
    read_public_knowledge_base_info,
)
from tests.helpers import make_chunk, write_chunks


def _public_info() -> PublicKnowledgeBaseInfo:
    return PublicKnowledgeBaseInfo(
        title="测试知识库",
        coverage="测试范围",
        topics=["主题"],
        capabilities=["能力"],
        guidance=["指引"],
        limitations=["边界"],
    )


def _manifest(source: Path) -> BookManifest:
    return BookManifest(
        schema_version="1.0",
        book_id="test-book",
        source_sha256=hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        source_page_count=4,
        sections=[
            BookSection(
                section_id="chapter-01",
                title="第1章",
                start_page=1,
                end_page=2,
            ),
            BookSection(
                section_id="table-of-contents",
                title="目录",
                start_page=3,
                end_page=3,
                include_in_index=False,
            ),
            BookSection(
                section_id="chapter-02",
                title="第2章",
                start_page=4,
                end_page=4,
            ),
        ],
        public_metadata=_public_info(),
    )


def test_full_book_manifest_declares_complete_text_scope():
    repository_root = Path(__file__).parents[1]
    manifest = read_book_manifest(
        repository_root
        / "configs"
        / "computer_networking"
        / "networking_top_down_8e.json"
    )

    assert manifest.source_page_count == 503
    assert manifest.sections[0].start_page == 5
    assert manifest.sections[-1].end_page == 501
    assert [
        section.section_id for section in manifest.sections
    ] == [
        "front-matter",
        "table-of-contents",
        "chapter-01",
        "chapter-02",
        "chapter-03",
        "chapter-04",
        "chapter-05",
        "chapter-06",
        "chapter-07",
        "chapter-08",
        "references",
    ]


def test_assemble_corpus_excludes_non_indexed_sections_and_renumbers(
    tmp_path: Path,
):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"book")
    manifest = _manifest(source)
    results: list[SectionBuildResult] = []
    for ordinal, section in enumerate(manifest.sections, start=1):
        chunks_path = tmp_path / f"{section.section_id}.jsonl"
        write_chunks(
            chunks_path,
            [
                make_chunk(
                    chunk_id=f"chunk-{section.section_id}",
                    ordinal=ordinal,
                )
            ],
        )
        results.append(
            SectionBuildResult(
                section=section,
                artifact_root=tmp_path / section.section_id,
                chunks_path=chunks_path,
                chunk_count=1,
                error_count=0,
                warning_count=ordinal,
            )
        )

    chunks_path = assemble_corpus(
        manifest=manifest,
        source_pdf=source,
        output_root=tmp_path / "output",
        section_results=results,
    )

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "chunk-chapter-01",
        "chunk-chapter-02",
    ]
    assert [chunk["ordinal"] for chunk in chunks] == [1, 2]
    generated_manifest = json.loads(
        (chunks_path.parent / "corpus-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated_manifest["indexed_chunk_count"] == 2
    assert generated_manifest["public_metadata"] == _public_info().model_dump()


def test_process_book_rejects_changed_source_before_writing(
    tmp_path: Path,
):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"book")
    manifest = _manifest(source).model_copy(
        update={"source_sha256": "a" * 64}
    )

    with pytest.raises(ValueError, match="SHA-256"):
        process_book(
            manifest=manifest,
            source_pdf=source,
            output_root=tmp_path / "output",
        )

    assert not (tmp_path / "output").exists()


def test_process_existing_docling_book_rebuilds_without_conversion(
    tmp_path: Path,
):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"book")
    manifest = _manifest(source)
    existing_root = tmp_path / "existing"
    for section in manifest.sections:
        section_root = (
            existing_root / "sections" / section.section_id
        )
        section_root.mkdir(parents=True)
        page = section.start_page
        document = {
            "pages": {
                str(page): {
                    "page_no": page,
                    "size": {"width": 500, "height": 700},
                }
            },
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "section_header",
                    "prov": [
                        {
                            "page_no": page,
                            "bbox": {
                                "l": 50,
                                "t": 650,
                                "r": 450,
                                "b": 620,
                                "coord_origin": "BOTTOMLEFT",
                            },
                        }
                    ],
                    "text": section.title,
                },
                {
                    "self_ref": "#/texts/1",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [
                        {
                            "page_no": page,
                            "bbox": {
                                "l": 50,
                                "t": 580,
                                "r": 450,
                                "b": 500,
                                "coord_origin": "BOTTOMLEFT",
                            },
                        }
                    ],
                    "text": f"{section.title} 正文。",
                },
            ],
        }
        (section_root / f"{section.section_id}.docling.json").write_text(
            json.dumps(document, ensure_ascii=False),
            encoding="utf-8",
        )
        (section_root / f"{section.section_id}.md").write_text(
            f"# {section.title}\n\n{section.title} 正文。\n",
            encoding="utf-8",
        )

    correction_root = tmp_path / "corrections"
    correction_root.mkdir()
    results = process_existing_docling_book(
        manifest=manifest,
        source_pdf=source,
        existing_docling_root=existing_root,
        correction_root=correction_root,
        output_root=(tmp_path / "rebuilt"),
    )

    assert len(results) == 3
    assert all(result.passed for result in results)
    chunks = [
        json.loads(line)
        for line in (
            tmp_path / "rebuilt" / "corpus" / "chunks.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(chunks) == 2


def test_public_metadata_loader_accepts_only_generated_public_section(
    tmp_path: Path,
):
    path = tmp_path / "corpus-manifest.json"
    path.write_text(
        json.dumps(
            {
                "public_metadata": _public_info().model_dump(),
                "collection": "must-not-be-public",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert read_public_knowledge_base_info(path) == _public_info()
