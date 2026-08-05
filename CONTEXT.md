# RAG Lab Context

## Glossary

- **KnowledgeChunk**: the source-grounded unit that is indexed and returned
  to a user.
- **SearchFilters**: storage-neutral constraints for document identity,
  heading prefix, and source-page interval.
- **Candidate**: an indexed Point that satisfies the active
  `SearchFilters`, before top-k selection.
- **Dense retrieval**: retrieval that embeds a query and searches stored
  vectors through `DenseRetriever`.
- **Retrieval runtime**: a command-line entry point that assembles concrete
  infrastructure and invokes a retriever; it does not implement retrieval
  policy.
