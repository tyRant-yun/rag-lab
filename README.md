# RAG Lab

RAG Lab is a local knowledge-base product for turning source documents
into traceable data that later retrieval components can consume.

The repository currently implements the document-normalization stage.
It converts Docling JSON plus its source PDF into ordered,
source-grounded `NormalizedBlock` records. Chunking, vector storage,
retrieval, agents, and API integration are intentionally outside the
current implementation.

## Product boundaries

- **Normalizer** owns reading order, text cleanup, structural type,
  heading path, page provenance, and optional image references.
- **Chunker** will consume only `NormalizedBlock` records and own block
  merging, target length, and overlap.
- **Retriever** will consume Chunker output and own search.

The Normalizer deliberately leaves adjacent and cross-page blocks
separate. Future Chunker code must not read Docling JSON or depend on
the review Markdown format.

## Python namespace

```text
src/rag_lab/
├── contracts/
│   └── blocks.py
└── normalization/
    ├── cli.py
    ├── models.py
    ├── normalizer.py
    └── serialization.py
```

`rag_lab.contracts` contains shared product contracts.
`rag_lab.normalization` contains the current source-specific ingestion
implementation and its quality report.

## NormalizedBlock contract

Each line in `blocks.jsonl` is one strict Pydantic
`NormalizedBlock` with exactly these fields:

- `block_id`: deterministic SHA-256 identity for the normalized block;
- `document_id`: SHA-256 identity derived from the source PDF bytes;
- `text` and `block_type`: normalized content and its structural role;
- `heading_path`: the active document and section hierarchy;
- `page_start` and `page_end`: inclusive PDF page provenance;
- `ordinal`: contiguous, one-based reading order;
- `source_path`: resolved source PDF path;
- `image_path`: optional normalized image path (`null` when absent);
- `normalization_version`: caller-controlled transformation version.

The output files are:

- `blocks.jsonl`: the structured downstream contract;
- `document.md`: a human-readable view generated from the same blocks;
- `normalization-report.json`: quality diagnostics.

Markdown is only a review view and is not a downstream interface.

## Run the Normalizer

```powershell
python -m rag_lab.normalization.cli `
  --input-json "path\to\document.docling.json" `
  --source "path\to\source.pdf" `
  --output "path\to\normalized" `
  --normalization-version "1.0.0"
```

After installing the project, the equivalent console command is:

```powershell
normalize-docling `
  --input-json "path\to\document.docling.json" `
  --source "path\to\source.pdf" `
  --output "path\to\normalized" `
  --normalization-version "1.0.0"
```

## Review and batch use

The current reading-order implementation is intentionally scoped to
single-column textbooks. It uses Docling provenance coordinates and
fails validation when required provenance is missing.

For batch processing, invoke the command once per source document and
write each result to its own output directory. Review
`pages_requiring_review` in `normalization-report.json` before accepting
the output. The report flags source pages containing a Docling
`section_header` that was conservatively downgraded to normal text.
