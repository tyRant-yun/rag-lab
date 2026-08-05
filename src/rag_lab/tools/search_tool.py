from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rag_lab.contracts import (
    SearchFilters,
    SearchHit,
    SearchResult,
)
from rag_lab.embeddings import (
    OllamaEmbeddingError,
)
from rag_lab.vector_store import (
    QdrantVectorStoreError,
)
from rag_lab.retrieval.factory import (
    RetrieverName,
)


class RetrievalSearcher(Protocol):
    """Minimal retriever interface used by the search tool."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        ...


class SearchKnowledgeArguments(BaseModel):
    """Arguments accepted by the search_knowledge tool."""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    retriever: RetrieverName = "rerank"
    document_ids: list[str] | None = None
    heading_prefix: list[str] | None = None
    page_start: int | None = Field(
        default=None,
        ge=1,
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_page_range(self) -> SearchKnowledgeArguments:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must not precede page_start"
            )

        return self


def serialize_hit(
    hit: SearchHit,
) -> dict[str, object]:
    """Serialize one hit into a bounded, agent-friendly dict."""

    chunk = hit.chunk

    return {
        "rank": hit.rank,
        "score": hit.score,
        "retriever": hit.retriever,
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "heading_path": list(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }


class SearchKnowledgeTool:
    """Stable, validated retrieval tool for LLM agents.

    The model only proposes arguments; the backend validates them with
    Pydantic, runs the whitelisted retriever and returns a bounded,
    serialized result.  Tools never grant the model arbitrary execution.
    """

    TOOL_NAME = "search_knowledge"
    DESCRIPTION = (
        "Search the local knowledge base and return the top matching "
        "source-grounded chunks with scores, headings and page "
        "references. Use this tool when an answer should be grounded "
        "in the knowledge base."
    )

    def __init__(
        self,
        *,
        retrievers: Mapping[
            str,
            RetrievalSearcher,
        ],
    ) -> None:
        if not retrievers:
            raise ValueError(
                "retrievers cannot be empty"
            )

        self._retrievers = dict(retrievers)

    @classmethod
    def openai_schema(cls) -> dict[str, object]:
        """OpenAI function-calling schema for this tool."""

        return {
            "type": "function",
            "function": {
                "name": cls.TOOL_NAME,
                "description": cls.DESCRIPTION,
                "parameters": (
                    SearchKnowledgeArguments
                    .model_json_schema()
                ),
            },
        }

    def execute(
        self,
        arguments: SearchKnowledgeArguments,
    ) -> dict[str, object]:
        """Run validated arguments and return a bounded result dict."""

        retriever = self._retrievers.get(
            arguments.retriever
        )

        if retriever is None:
            return {
                "success": False,
                "tool": self.TOOL_NAME,
                "error": (
                    "unknown retriever: "
                    f"{arguments.retriever}"
                ),
            }

        try:
            filters = SearchFilters(
                document_ids=(
                    arguments.document_ids
                ),
                heading_prefix=(
                    arguments.heading_prefix
                ),
                page_start=arguments.page_start,
                page_end=arguments.page_end,
            )

            result = retriever.search(
                arguments.query,
                top_k=arguments.top_k,
                filters=filters,
            )
        except (
            TypeError,
            ValueError,
            ValidationError,
            OllamaEmbeddingError,
            QdrantVectorStoreError,
        ) as error:
            return {
                "success": False,
                "tool": self.TOOL_NAME,
                "error": str(error),
            }

        return {
            "success": True,
            "tool": self.TOOL_NAME,
            "retriever": result.retriever,
            "index_version": result.index_version,
            "candidate_count": result.candidate_count,
            "count": len(result.hits),
            "hits": [
                serialize_hit(hit)
                for hit in result.hits
            ],
        }

    def execute_raw(
        self,
        arguments: Mapping[str, object],
    ) -> dict[str, object]:
        """Validate raw model-produced arguments and execute."""

        try:
            validated = (
                SearchKnowledgeArguments
                .model_validate(
                    dict(arguments)
                )
            )
        except ValidationError as error:
            return {
                "success": False,
                "tool": self.TOOL_NAME,
                "error": (
                    "invalid arguments: "
                    f"{error}"
                ),
            }

        return self.execute(validated)
