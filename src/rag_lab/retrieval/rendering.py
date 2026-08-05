from __future__ import annotations

from rag_lab.contracts import SearchResult


def render_human_result(
    result: SearchResult,
) -> str:
    """Render one SearchResult for human-readable CLI output."""

    lines = [
        f"Query: {result.query}",
        f"Retriever: {result.retriever}",
        f"Index version: {result.index_version}",
        f"Candidates: {result.candidate_count}",
        f"Hits: {len(result.hits)}",
        f"Elapsed: {result.elapsed_ms:.2f} ms",
    ]

    if not result.hits:
        lines.extend(
            [
                "",
                "No matching chunks.",
            ]
        )
        return "\n".join(lines)

    for hit in result.hits:
        chunk = hit.chunk
        heading = " > ".join(
            chunk.heading_path
        )

        lines.extend(
            [
                "",
                (
                    f"[{hit.rank}] "
                    f"score={hit.score:.6f}"
                ),
                f"chunk_id={chunk.chunk_id}",
                (
                    f"pages={chunk.page_start}-"
                    f"{chunk.page_end}"
                ),
                f"heading={heading}",
                f"source={chunk.source_path}",
                "",
                chunk.content,
            ]
        )

    return "\n".join(lines)
