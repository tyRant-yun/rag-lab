"""Reproducible, source-grounded book ingestion helpers."""

from rag_lab.ingestion.manifest import (
    BookManifest,
    BookSection,
    read_book_manifest,
)
from rag_lab.ingestion.pipeline import (
    assemble_corpus,
    process_book,
    process_existing_docling_book,
)

__all__ = [
    "BookManifest",
    "BookSection",
    "assemble_corpus",
    "process_book",
    "process_existing_docling_book",
    "read_book_manifest",
]
