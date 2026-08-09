# Private RAG Lab runtime

The image contains code only. It intentionally excludes PDFs, `chunks.jsonl`,
local paths, and Qdrant data. Mount an approved `chunks.jsonl` read-only at
`RAG_CHUNKS_PATH`; do not place source PDFs or unapproved textbook-derived
artifacts in the image or a public registry.

`serve-api` reads every `RAG_*` setting from the environment. Its debug routes
are disabled by default. Enable `--enable-debug-routes` only for local,
loopback development; production must keep it disabled regardless of bind
address.

Health endpoints have separate meanings:

- `/health/live` only confirms that the API process is running.
- `/health/ready` verifies the already-loaded chunks, Qdrant reachability and
  collection compatibility, plus that the configured Ollama endpoint lists the
  configured model. It does not run an embedding or retrieval.

The search runtime only reads an existing Qdrant collection. It never creates,
migrates, deletes, or overwrites a collection. Before deployment, record the
approved chunk artifact SHA-256, collection name, vector dimension, embedding
model/version, and expected point count. A collection with different dimensions
is rejected by readiness and search.

The compose bundle is maintained by the Interview Agent repository because it
owns the browser-facing product. See its `deploy/README.md` for the network,
volume, backup, and corpus-license release gates.
