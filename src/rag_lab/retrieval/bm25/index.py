from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag_lab.contracts import KnowledgeChunk
from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)


class BM25Index:
    """Immutable in-memory BM25 index over KnowledgeChunk objects."""

    def __init__(
        self,
        *,
        chunks: Sequence[KnowledgeChunk],
        analyzer: LexicalAnalyzer,
    ) -> None:
        self._chunks = tuple(chunks)
        self._analyzer = analyzer

        if not self._chunks:
            raise ValueError("chunks cannot be empty")

        chunk_ids = [
            chunk.chunk_id
            for chunk in self._chunks
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                "chunk IDs cannot contain duplicates"
            )

        tokenized_corpus: list[tuple[str, ...]] = []

        for chunk in self._chunks:
            terms = analyzer.analyze(
                chunk.index_text
            )

            if not terms:
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} "
                    "produced no lexical terms"
                )

            tokenized_corpus.append(terms)

        self._tokenized_corpus = tuple(
            tokenized_corpus
        )

        self._engine = BM25Okapi(
            [
                list(terms)
                for terms in self._tokenized_corpus
            ]
        )

        self._index_version = (
            self._build_index_version()
        )

    @property
    def chunks(self) -> tuple[KnowledgeChunk, ...]:
        return self._chunks

    @property
    def tokenized_corpus(
        self,
    ) -> tuple[tuple[str, ...], ...]:
        return self._tokenized_corpus

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def index_version(self) -> str:
        return self._index_version

    def score_query(
        self,
        query: str,
    ) -> tuple[float, ...]:
        query_terms = self._analyzer.analyze(
            query
        )

        if not query_terms:
            return tuple(
                0.0
                for _ in self._chunks
            )

        scores = self._engine.get_scores(
            list(query_terms)
        )

        return tuple(
            float(score)
            for score in scores
        )

    def _build_index_version(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"bm25-index-v1\0")

        for chunk, terms in zip(
            self._chunks,
            self._tokenized_corpus,
            strict=True,
        ):
            digest.update(
                chunk.chunk_id.encode("utf-8")
            )
            digest.update(b"\0")

            for term in terms:
                digest.update(
                    term.encode("utf-8")
                )
                digest.update(b"\0")

            digest.update(b"\xff")

        return f"bm25-v1:{digest.hexdigest()}"
