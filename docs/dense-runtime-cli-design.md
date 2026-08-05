# Dense Runtime CLI Design

**Status:** Proposed — implementation starts after boundary confirmation.

## Goal

Provide a `search-dense` command that runs an existing Qdrant collection
through the existing `DenseRetriever` and presents one `SearchResult` as text
or JSON.

The command is a runtime adapter, not a second dense-retrieval
implementation.

## Scope

The implementation will add:

- the `search-dense` console-script entry point;
- `rag_lab.retrieval.dense.cli` for argument parsing, infrastructure
  assembly, error-to-exit-code conversion, and rendering;
- Fake Provider/Fake Store tests that exercise the CLI's public `main`
  seam without Ollama or Qdrant; and
- one manual real-service smoke command after all unit tests pass.

It will not add Dense evaluation, Hybrid/RRF, API or Agent integration,
incremental indexing, deletion, collection migration, or changes to BM25
behavior.

## Command surface

```text
search-dense \
  --collection <name> \
  --query <text> \
  [--top-k 5] \
  [--url http://localhost:6333] \
  [--model qwen3-embedding:0.6b] \
  [--host http://localhost:11434] \
  [--dimensions 1024] \
  [--embedding-timeout-seconds 60] \
  [--qdrant-timeout-seconds 10] \
  [--document-id <id> ...] \
  [--heading <component> ...] \
  [--page-start <page>] \
  [--page-end <page>] \
  [--json]
```

`--collection` and `--query` are required. The four filter options map
directly to every `SearchFilters` field. Repeated `--document-id` and
`--heading` preserve their argument order.

The command deliberately has no `--chunks` option: an existing Qdrant
collection is the runtime source of indexed `KnowledgeChunk` records. It does
not create, migrate, delete, or populate a collection.

## Composition boundary

`main()` will:

1. parse CLI arguments;
2. build `OllamaEmbeddingProvider` from model, host, dimensions, and timeout;
3. build `QdrantVectorStore` from URL, collection, dimensions, and timeout;
4. construct `DenseRetriever(provider=..., store=...)`;
5. construct one `SearchFilters` instance from the CLI filters;
6. call only `DenseRetriever.search(query, top_k=..., filters=...)`; and
7. render its `SearchResult`.

The CLI will not call `embed_query`, `count`, or `search` on infrastructure
objects directly. It will not call `ensure_collection`, because search must
not mutate runtime storage.

The implementation will use injectable provider and store factories, matching
the established `index-qdrant` test seam. The default store factory may reuse
the existing Qdrant construction helper; no new cross-retriever or Hybrid
abstraction is needed.

## Output and failure behavior

Text output will preserve the existing BM25 result layout: query, retriever,
index version, candidate count, hit count, elapsed time, then source-grounded
hit details. `--json` will emit `SearchResult.to_dict()` with UTF-8 Chinese
text preserved and no extra stdout output.

Expected configuration, validation, Ollama, Qdrant, and filesystem failures
will write `error: <message>` to stderr and return exit code `2`. Successful
search returns `0`. Stable exceptions from `DenseRetriever` are not silently
converted into an empty result.

## Test seams and acceptance checks

`tests/test_dense_cli.py` will use fakes injected into `main()` to verify:

- parser-to-factory configuration transport;
- query embedding occurs through `DenseRetriever` exactly once;
- count and search receive the same `SearchFilters` values;
- text and JSON output expose the returned `SearchResult`;
- document, heading, and page filters reach the store;
- empty results and expected errors have the established CLI behavior; and
- no unit test opens an Ollama or Qdrant connection.

The final manual smoke check will use a pre-indexed local Qdrant collection
and local Ollama model. Its report will record the exact command, result hit
IDs, `candidate_count`, and `index_version`; it is validation evidence rather
than an automated unit test.
