# RAG Lab Architecture

## Dependency direction

`contracts` is the lowest-level shared vocabulary. Normalization produces
`NormalizedBlock`; Chunking consumes blocks and produces `KnowledgeChunk`;
retrieval, evaluation, API and tools consume chunks and search contracts.
Concrete embedding and vector-store integrations sit behind their respective
provider contracts.

```text
Docling → Normalizer → NormalizedBlock → Chunker → KnowledgeChunk
                                           ↓
                         BM25 / Dense → Hybrid → Rerank
                                           ↓
                       CLI / Evaluation / API / SearchKnowledgeTool
```

## Contracts and data paths

- `NormalizedBlock`, `KnowledgeChunk`, `SearchResult`, embedding and
  vector-store contracts are the public machine interfaces. Markdown is only
  an inspection view.
- The write path is conversion → normalization → chunking → optional embedding
  and indexing. Runtime retrieval paths are read-only and must not create,
  migrate or mutate a Qdrant collection.
- `RetrievalComponents` is the one shared assembly boundary used by retrieval
  CLIs, the API and `RetrievalToolset`; those adapters do not reimplement
  retrieval policy.

## Responsibility boundaries

- Converter/Docling owns PDF parsing, OCR, layout, images, tables and upstream
  formula recognition.
- Normalizer owns deterministic text cleanup, source order, block types,
  headings, pages, image links and reviewed correction overlays.
- A correction overlay is data, not an edited artifact: each operation must
  name its Docling source references and adjacent text anchors. The normalizer
  validates those anchors before rebuilding blocks and their IDs; non-indexable
  figure labels retain provenance while the chunker excludes them from search.
- Chunker consumes only normalized blocks. It forms retrieval units, preserves
  provenance and enforces size/overlap rules; it does not repair upstream text
  or layout errors.
- Retrieval ranks existing chunks. API and `search_knowledge` return grounded
  search results; neither is a complete LLM Agent runtime or final-answer
  generator.
