"""One-write-path orchestration for reproducible multi-section books."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rag_lab.chunking import (
    ChunkingConfig,
    chunk_normalized_blocks,
)
from rag_lab.chunking.serialization import (
    read_normalized_blocks_jsonl,
    write_chunking_outputs,
)
from rag_lab.contracts import KnowledgeChunk
from rag_lab.ingestion.manifest import (
    BookManifest,
    BookSection,
)
from rag_lab.normalization import (
    normalize_docling_document,
)
from rag_lab.normalization.corrections import (
    CorrectionOverlay,
    read_correction_overlay,
)
from rag_lab.normalization.serialization import (
    write_normalization_outputs,
)
from rag_lab.quality import audit_artifacts
from rag_lab.quality.models import (
    ArtifactQualityInputs,
    AuditConfig,
)
from rag_lab.quality.serialization import (
    write_artifact_quality_report,
)


NORMALIZATION_VERSION = "2.0.0"
CHUNKING_VERSION = "2.0.0"
MAX_CHARS = 1200
OVERLAP_CHARS = 120


@dataclass(frozen=True)
class SectionBuildResult:
    """Paths and quality outcome for one independently auditable section."""

    section: BookSection
    artifact_root: Path
    chunks_path: Path
    chunk_count: int
    error_count: int
    warning_count: int

    @property
    def passed(self) -> bool:
        return self.error_count == 0


def process_book(
    *,
    manifest: BookManifest,
    source_pdf: Path,
    output_root: Path,
    section_ids: Sequence[str] | None = None,
) -> tuple[SectionBuildResult, ...]:
    """Convert, normalize, chunk and audit selected book sections.

    Output directories must be new. This intentionally refuses in-place
    regeneration so a prior accepted artifact cannot be silently overwritten.
    """

    _validate_source(source_pdf=source_pdf, manifest=manifest)
    selected_sections = _select_sections(
        manifest=manifest,
        section_ids=section_ids,
    )
    if output_root.exists():
        raise ValueError(
            "output_root already exists; choose a new artifact version"
        )

    results: list[SectionBuildResult] = []
    for section in selected_sections:
        results.append(
            _process_section(
                section=section,
                source_pdf=source_pdf,
                output_root=output_root,
            )
        )

    if any(not result.passed for result in results):
        raise ValueError(
            "quality audit reported errors; corpus assembly was not run"
        )

    if len(selected_sections) == len(manifest.sections):
        assemble_corpus(
            manifest=manifest,
            source_pdf=source_pdf,
            output_root=output_root,
            section_results=results,
        )

    return tuple(results)


def process_existing_docling_book(
    *,
    manifest: BookManifest,
    source_pdf: Path,
    existing_docling_root: Path,
    correction_root: Path,
    output_root: Path,
) -> tuple[SectionBuildResult, ...]:
    """Rebuild a full corpus from an immutable prior Docling conversion.

    This path deliberately copies the source conversion into a new artifact
    version, then reruns normalization, chunking and quality gates. It never
    mutates the prior conversion or hand-edits its generated files.
    """

    _validate_source(source_pdf=source_pdf, manifest=manifest)
    if output_root.exists():
        raise ValueError(
            "output_root already exists; choose a new artifact version"
        )
    if not existing_docling_root.is_dir():
        raise ValueError("existing_docling_root must be a directory")
    if not correction_root.is_dir():
        raise ValueError("correction_root must be a directory")

    results = [
        _process_existing_docling_section(
            section=section,
            source_pdf=source_pdf,
            existing_docling_root=existing_docling_root,
            correction_root=correction_root,
            output_root=output_root,
        )
        for section in manifest.sections
    ]
    if any(not result.passed for result in results):
        raise ValueError(
            "quality audit reported errors; corpus assembly was not run"
        )

    assemble_corpus(
        manifest=manifest,
        source_pdf=source_pdf,
        output_root=output_root,
        section_results=results,
    )
    return tuple(results)


def assemble_corpus(
    *,
    manifest: BookManifest,
    source_pdf: Path,
    output_root: Path,
    section_results: Sequence[SectionBuildResult],
) -> Path:
    """Create the one corpus JSONL consumed by retrieval and indexing."""

    _validate_source(source_pdf=source_pdf, manifest=manifest)
    expected_ids = {
        section.section_id for section in manifest.sections
    }
    actual_ids = {
        result.section.section_id for result in section_results
    }
    if actual_ids != expected_ids:
        raise ValueError(
            "corpus assembly requires a result for every manifest section"
        )
    if any(not result.passed for result in section_results):
        raise ValueError(
            "cannot assemble a corpus with failing section audits"
        )

    chunks: list[KnowledgeChunk] = []
    seen_chunk_ids: set[str] = set()
    for result in section_results:
        if not result.section.include_in_index:
            continue
        for chunk in _read_chunks(result.chunks_path):
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    "duplicate chunk_id across book sections: "
                    f"{chunk.chunk_id}"
                )
            seen_chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ValueError("book manifest selects no searchable chunks")

    corpus_directory = output_root / "corpus"
    corpus_directory.mkdir(parents=True, exist_ok=False)
    indexed_chunks = tuple(
        chunk.model_copy(update={"ordinal": ordinal})
        for ordinal, chunk in enumerate(chunks, start=1)
    )
    chunks_path = corpus_directory / "chunks.jsonl"
    _write_chunks(indexed_chunks, chunks_path)

    corpus_manifest = {
        "schema_version": "1.0",
        "book_id": manifest.book_id,
        "source_sha256": manifest.source_sha256.lower(),
        "source_page_count": manifest.source_page_count,
        "public_metadata": manifest.public_metadata.model_dump(
            mode="json"
        ),
        "included_sections": [
            {
                "section_id": result.section.section_id,
                "title": result.section.title,
                "start_page": result.section.start_page,
                "end_page": result.section.end_page,
                "include_in_index": result.section.include_in_index,
                "chunk_count": result.chunk_count,
                "warning_count": result.warning_count,
            }
            for result in section_results
        ],
        "indexed_chunk_count": len(indexed_chunks),
        "normalization_version": NORMALIZATION_VERSION,
        "chunking_version": CHUNKING_VERSION,
    }
    (corpus_directory / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return chunks_path


def _process_section(
    *,
    section: BookSection,
    source_pdf: Path,
    output_root: Path,
) -> SectionBuildResult:
    section_root = output_root / "sections" / section.section_id
    section_root.mkdir(parents=True, exist_ok=False)
    docling_markdown, docling_json = _convert_section(
        section=section,
        source_pdf=source_pdf,
        output_directory=section_root,
    )
    return _normalize_chunk_and_audit_section(
        section=section,
        source_pdf=source_pdf,
        section_root=section_root,
        docling_markdown=docling_markdown,
        docling_json=docling_json,
        artifact_directory=section_root,
        correction_overlay=None,
    )


def _process_existing_docling_section(
    *,
    section: BookSection,
    source_pdf: Path,
    existing_docling_root: Path,
    correction_root: Path,
    output_root: Path,
) -> SectionBuildResult:
    source_section = (
        existing_docling_root / "sections" / section.section_id
    )
    source_markdown = source_section / f"{section.section_id}.md"
    source_json = source_section / f"{section.section_id}.docling.json"
    if not source_markdown.is_file() or not source_json.is_file():
        raise ValueError(
            "missing prior Docling conversion for section: "
            f"{section.section_id}"
        )

    section_root = output_root / "sections" / section.section_id
    section_root.mkdir(parents=True, exist_ok=False)
    docling_markdown = section_root / source_markdown.name
    docling_json = section_root / source_json.name
    shutil.copy2(source_markdown, docling_markdown)
    shutil.copy2(source_json, docling_json)

    overlay_path = correction_root / f"{section.section_id}.json"
    correction_overlay = (
        read_correction_overlay(overlay_path)
        if overlay_path.is_file()
        else None
    )
    return _normalize_chunk_and_audit_section(
        section=section,
        source_pdf=source_pdf,
        section_root=section_root,
        docling_markdown=docling_markdown,
        docling_json=docling_json,
        artifact_directory=source_section,
        correction_overlay=correction_overlay,
    )


def _normalize_chunk_and_audit_section(
    *,
    section: BookSection,
    source_pdf: Path,
    section_root: Path,
    docling_markdown: Path,
    docling_json: Path,
    artifact_directory: Path,
    correction_overlay: CorrectionOverlay | None,
) -> SectionBuildResult:
    normalized_directory = section_root / "normalized"
    docling_document = json.loads(docling_json.read_text(encoding="utf-8"))
    normalization = normalize_docling_document(
        docling_document=docling_document,
        source_path=source_pdf,
        normalization_version=NORMALIZATION_VERSION,
        artifact_directory=artifact_directory,
        correction_overlay=correction_overlay,
    )
    write_normalization_outputs(
        result=normalization,
        output_directory=normalized_directory,
        asset_source_directory=artifact_directory,
    )

    chunked_directory = (
        section_root / f"chunked-max{MAX_CHARS}-overlap{OVERLAP_CHARS}"
    )
    chunking = chunk_normalized_blocks(
        blocks=read_normalized_blocks_jsonl(
            normalized_directory / "blocks.jsonl"
        ),
        config=ChunkingConfig(
            max_chars=MAX_CHARS,
            overlap_chars=OVERLAP_CHARS,
            chunking_version=CHUNKING_VERSION,
        ),
    )
    write_chunking_outputs(
        result=chunking,
        output_directory=chunked_directory,
    )

    quality_report = audit_artifacts(
        inputs=ArtifactQualityInputs(
            docling_markdown=docling_markdown,
            normalization_report=(
                normalized_directory / "normalization-report.json"
            ),
            blocks=normalized_directory / "blocks.jsonl",
            chunking_report=(
                chunked_directory / "chunking-report.json"
            ),
            chunks=chunked_directory / "chunks.jsonl",
        ),
        config=AuditConfig(max_chunk_chars=MAX_CHARS),
    )
    write_artifact_quality_report(
        report=quality_report,
        output_path=section_root / "artifact-quality-report.json",
    )
    return SectionBuildResult(
        section=section,
        artifact_root=section_root,
        chunks_path=chunked_directory / "chunks.jsonl",
        chunk_count=len(chunking.chunks),
        error_count=quality_report.error_count,
        warning_count=quality_report.warning_count,
    )


def _convert_section(
    *,
    section: BookSection,
    source_pdf: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Convert one range, importing the optional Docling dependency lazily."""

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )
        from docling_core.types.doc import ImageRefMode
    except ImportError as error:
        raise RuntimeError(
            "Docling is required for conversion; install rag-lab[conversion]"
        ) from error

    assets_directory = output_directory / "assets"
    assets_directory.mkdir()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    # Formula markers are quality-gate errors.  Enable Docling's dedicated
    # formula model so formulas are represented as equations instead of
    # relying on a count-only restoration claim.
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 1.5
    # Docling enables torch.compile by default. On a local Windows machine
    # without MSVC this makes ordinary layout inference fail before reading
    # the PDF. Layout quality does not require that optional optimization.
    pipeline_options.layout_options.engine_options.compile_model = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    result = converter.convert(
        source_pdf,
        page_range=(section.start_page, section.end_page),
    )
    markdown_path = output_directory / f"{section.section_id}.md"
    json_path = output_directory / f"{section.section_id}.docling.json"
    result.document.save_as_markdown(
        markdown_path,
        artifacts_dir=assets_directory,
        image_mode=ImageRefMode.REFERENCED,
        page_break_placeholder="\n\n<!-- page-break -->\n\n",
    )
    result.document.save_as_json(
        json_path,
        artifacts_dir=assets_directory,
        image_mode=ImageRefMode.REFERENCED,
    )
    return markdown_path, json_path


def _validate_source(
    *,
    source_pdf: Path,
    manifest: BookManifest,
) -> None:
    if not source_pdf.is_file():
        raise FileNotFoundError(f"source PDF not found: {source_pdf}")
    actual_hash = _sha256(source_pdf)
    if actual_hash.lower() != manifest.source_sha256.lower():
        raise ValueError(
            "source PDF SHA-256 does not match the book manifest"
        )


def _select_sections(
    *,
    manifest: BookManifest,
    section_ids: Sequence[str] | None,
) -> tuple[BookSection, ...]:
    if not section_ids:
        return tuple(manifest.sections)
    requested = set(section_ids)
    known = {
        section.section_id: section
        for section in manifest.sections
    }
    unknown = sorted(requested - set(known))
    if unknown:
        raise ValueError(
            "unknown section IDs: " + ", ".join(unknown)
        )
    return tuple(
        section
        for section in manifest.sections
        if section.section_id in requested
    )


def _read_chunks(path: Path) -> tuple[KnowledgeChunk, ...]:
    chunks: list[KnowledgeChunk] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            raise ValueError(
                f"{path}: empty JSONL record at line {line_number}"
            )
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}: invalid JSONL record at line {line_number}"
            ) from error
        chunks.append(
            KnowledgeChunk.model_validate(payload, strict=True)
        )
    return tuple(chunks)


def _write_chunks(
    chunks: Sequence[KnowledgeChunk],
    path: Path,
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                chunk.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
