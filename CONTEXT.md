# RAG Lab Context

## Glossary

- **KnowledgeChunk**: the source-grounded unit that is indexed and returned
  to a user.
- **SearchFilters**: storage-neutral constraints for document identity,
  heading prefix, and source-page interval.
- **Candidate**: an indexed Point that satisfies the active
  `SearchFilters`, before a retriever's top-k selection. Its population is
  retriever-specific: for BM25 and Dense it is that retriever's filtered
  index; for Hybrid, `candidate_count` is the union of the two truncated
  input rankings, not the full corpus.
- **Dense retrieval**: retrieval that embeds a query and searches stored
  vectors through `DenseRetriever`.
- **Retrieval runtime**: a command-line entry point that assembles concrete
  infrastructure and invokes a retriever; it does not implement retrieval
  policy.
- **HybridRetriever**: combines BM25 and Dense rankings with Reciprocal Rank
  Fusion (RRF).
- **Reciprocal Rank Fusion (RRF)**: a deterministic rank-fusion method that
  sums `1 / (k + rank)` for each retriever result.
- **Reranker**: transforms an already retrieved candidate ranking without
  taking responsibility for base retrieval timing.
- **RerankedRetriever**: owns the full retrieve-then-rerank operation; its
  `elapsed_ms` includes both stages.
- **RetrievalComponents**: the shared assembly of analyzer, BM25 and Dense
  infrastructure used by CLIs, API and tools.
- **Retrieval API**: an HTTP entry point for retrieval results; it does not
  generate a final answer.
- **SearchKnowledgeTool**: the validated `search_knowledge` function schema
  and execution boundary exposed to an Agent.
- **RetrievalToolset**: assembles SearchKnowledgeTool over RetrievalComponents.
- **Artifact Quality Issue**: one deterministic, source-grounded finding from
  the artifact audit, with evidence and a remediation hint.
- **Correction Overlay**: a versioned, declarative set of narrowly scoped
  content corrections applied during reproducible normalization; it never
  hand-edits final JSONL artifacts.
