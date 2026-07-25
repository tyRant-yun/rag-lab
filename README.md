# Knowledge Normalizer

Convert Docling JSON into ordered, source-grounded `NormalizedBlock`
records.

The project deliberately stops before RAG chunking and indexing. Its
outputs are:

- `blocks.jsonl`: the structured downstream contract;
- `document.md`: a human-readable view generated from the same blocks;
- `normalization-report.json`: quality diagnostics.

Each line in `blocks.jsonl` is one Pydantic `NormalizedBlock` with
exactly these fields:

- `block_id`: deterministic SHA-256 identity for the normalized block;
- `document_id`: SHA-256 identity derived from the source PDF bytes;
- `text` and `block_type`: normalized content and its structural role;
- `heading_path`: the active document/section hierarchy;
- `page_start` and `page_end`: inclusive PDF page provenance;
- `ordinal`: contiguous, one-based reading order;
- `source_path`: resolved source PDF path;
- `image_path`: optional normalized image path (`null` when absent);
- `normalization_version`: caller-controlled transformation version.

`document.md` is rendered from those same blocks, so it is a review view
rather than a second source of truth. HTML comments expose each block's
ordinal, type, and page range.

## Pipeline boundary

- **Normalizer** owns reading order, text cleanup, structural type,
  heading path, page provenance, and optional image references.
- **Chunker** consumes only `NormalizedBlock` records and owns block
  merging, target length, and overlap.
- **Retriever** consumes the Chunker's output and owns search.

The Chunker must not read Docling JSON or depend on the review Markdown
format. Neither Chunker nor Retriever is implemented in this package.

## Usage

```powershell
python -m knowledge_normalizer.cli `
  --input-json "path\to\document.docling.json" `
  --source "path\to\source.pdf" `
  --output "path\to\normalized" `
  --normalization-version "1.0.0"
```

The current reading-order implementation is intentionally scoped to
single-column textbooks. It uses Docling provenance coordinates and
fails validation when required provenance is missing.

For batch processing, invoke the command once per source document and
write each result to its own output directory. Review
`pages_requiring_review` in `normalization-report.json` before accepting
the output. In particular, the report flags source pages containing a
Docling `section_header` that was conservatively downgraded to normal
text.
