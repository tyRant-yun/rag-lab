from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
    SearchFilters,
    SearchHit,
    SearchResult,
    VectorMatch,
    VectorRecord,
    VectorWriteReport,
)
from rag_lab.evaluation import RetrievalEvaluationCase
from rag_lab.evaluation.dense_cli import main
from rag_lab.vector_store import QdrantVectorStore


def make_chunk(
    *,
    chunk_id: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=f"正文 {chunk_id}",
        index_text=f"索引正文 {chunk_id}",
        heading_path=["第一章"],
        page_start=ordinal,
        page_end=ordinal,
        ordinal=ordinal,
        block_ids=[f"block-{ordinal}"],
        source_path="book.pdf",
        content_hash=f"hash-{ordinal}",
        normalization_version="normalizer-v1",
        chunking_version="chunker-v1",
    )


def write_cases(
    path: Path,
    cases: Sequence[RetrievalEvaluationCase],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                case.to_dict(),
                ensure_ascii=False,
            )
            for case in cases
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class FakeEmbeddingProvider:
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def embedding_version(self) -> str:
        return "fake:fake-model:2:v1"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        del texts
        raise AssertionError(
            "evaluate-dense must not embed documents"
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        del text
        raise AssertionError(
            "FakeRetriever owns evaluation search calls"
        )


class FakeVectorStore:
    @property
    def collection_name(self) -> str:
        return "dense-test"

    @property
    def dimensions(self) -> int:
        return 2

    def ensure_collection(self) -> None:
        raise AssertionError(
            "evaluate-dense must not create collections"
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorWriteReport:
        del records
        raise AssertionError(
            "evaluate-dense must not index records"
        )

    def count(
        self,
        *,
        filters: SearchFilters | None = None,
    ) -> int:
        del filters

        raise AssertionError(
            "FakeRetriever owns evaluation search calls"
        )

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[VectorMatch]:
        del vector, top_k, filters
        raise AssertionError(
            "FakeRetriever owns evaluation search calls"
        )


class LocalModeEmbeddingProvider(
    FakeEmbeddingProvider
):
    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        del text
        return EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=self.dimensions,
        )


class FakeRetriever:
    def __init__(
        self,
        *,
        chunks_by_query: dict[str, KnowledgeChunk],
    ) -> None:
        self._chunks_by_query = chunks_by_query
        self.calls: list[
            tuple[
                str,
                int,
                SearchFilters | None,
            ]
        ] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        self.calls.append((query, top_k, filters))
        chunk = self._chunks_by_query[query]

        return SearchResult(
            query=query,
            hits=[
                SearchHit(
                    chunk=chunk,
                    score=0.9,
                    rank=1,
                    retriever="dense",
                )
            ],
            candidate_count=2,
            elapsed_ms=1.0,
            retriever="dense",
            index_version=(
                "dense-v1:dense-test:"
                "fake:fake-model:2:v1"
            ),
        )


def test_writes_human_report_and_routes_configuration(
    tmp_path: Path,
    capsys,
):
    tcp_chunk = make_chunk(
        chunk_id="chunk-tcp",
        ordinal=1,
    )
    dns_chunk = make_chunk(
        chunk_id="chunk-dns",
        ordinal=2,
    )
    tcp_filters = SearchFilters(
        document_ids=["document-a"],
        heading_prefix=["第一章"],
        page_start=1,
        page_end=2,
    )
    cases_path = tmp_path / "cases.jsonl"
    write_cases(
        cases_path,
        [
            RetrievalEvaluationCase(
                case_id="tcp-case",
                query="TCP",
                relevant_chunk_ids=["chunk-tcp"],
                filters=tcp_filters,
            ),
            RetrievalEvaluationCase(
                case_id="dns-case",
                query="DNS",
                relevant_chunk_ids=["chunk-dns"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    retriever = FakeRetriever(
        chunks_by_query={
            "TCP": tcp_chunk,
            "DNS": dns_chunk,
        },
    )
    provider_config: list[dict[str, object]] = []
    store_config: list[dict[str, object]] = []
    retriever_config: list[tuple[object, object]] = []

    def provider_factory(
        **kwargs: object,
    ) -> FakeEmbeddingProvider:
        provider_config.append(kwargs)
        return provider

    def store_factory(
        **kwargs: object,
    ) -> FakeVectorStore:
        store_config.append(kwargs)
        return store

    def retriever_factory(
        *,
        provider: object,
        store: object,
    ) -> FakeRetriever:
        retriever_config.append((provider, store))
        return retriever

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--dataset-id",
            "test-dataset",
            "--collection",
            "dense-test",
            "--top-k",
            "3",
            "--url",
            "http://qdrant.test:6333",
            "--model",
            "fake-model",
            "--host",
            "http://ollama.test:11434",
            "--dimensions",
            "2",
            "--embedding-timeout-seconds",
            "12.5",
            "--qdrant-timeout-seconds",
            "9",
        ],
        provider_factory=provider_factory,
        store_factory=store_factory,
        retriever_factory=retriever_factory,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert provider_config == [
        {
            "model_name": "fake-model",
            "dimensions": 2,
            "host": "http://ollama.test:11434",
            "timeout_seconds": 12.5,
        }
    ]
    assert store_config == [
        {
            "url": "http://qdrant.test:6333",
            "collection_name": "dense-test",
            "dimensions": 2,
            "timeout_seconds": 9,
        }
    ]
    assert retriever_config == [(provider, store)]
    assert retriever.calls == [
        ("TCP", 3, tcp_filters),
        ("DNS", 3, None),
    ]
    assert "Dataset: test-dataset" in captured.out
    assert "Retriever: dense" in captured.out
    assert (
        "Index version: dense-v1:dense-test:"
        "fake:fake-model:2:v1"
    ) in captured.out
    assert "Top K: 3" in captured.out
    assert "Cases: 2" in captured.out
    assert "Hit@3: 1.000000" in captured.out
    assert "[tcp-case]" in captured.out
    assert "[dns-case]" in captured.out


def test_writes_json_report(
    tmp_path: Path,
    capsys,
):
    chunk = make_chunk(
        chunk_id="chunk-json",
        ordinal=1,
    )
    cases_path = tmp_path / "cases.jsonl"
    write_cases(
        cases_path,
        [
            RetrievalEvaluationCase(
                case_id="json-case",
                query="JSON",
                relevant_chunk_ids=["chunk-json"],
            )
        ],
    )
    retriever = FakeRetriever(
        chunks_by_query={"JSON": chunk},
    )

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--dataset-id",
            "json-dataset",
            "--collection",
            "dense-test",
            "--top-k",
            "1",
            "--dimensions",
            "2",
            "--json",
        ],
        provider_factory=lambda **_: FakeEmbeddingProvider(),
        store_factory=lambda **_: FakeVectorStore(),
        retriever_factory=lambda **_: retriever,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["dataset_id"] == "json-dataset"
    assert payload["retriever"] == "dense"
    assert payload["index_version"] == (
        "dense-v1:dense-test:fake:fake-model:2:v1"
    )
    assert payload["top_k"] == 1
    assert payload["case_count"] == 1
    assert payload["case_results"][0][
        "first_relevant_rank"
    ] == 1


def test_reports_factory_error(
    tmp_path: Path,
    capsys,
):
    cases_path = tmp_path / "cases.jsonl"
    write_cases(
        cases_path,
        [
            RetrievalEvaluationCase(
                case_id="failure-case",
                query="failure",
                relevant_chunk_ids=["chunk-failure"],
            )
        ],
    )

    def failing_provider_factory(
        **kwargs: object,
    ) -> FakeEmbeddingProvider:
        del kwargs
        raise ValueError("provider setup failed")

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--dataset-id",
            "failure-dataset",
            "--collection",
            "dense-test",
            "--dimensions",
            "2",
        ],
        provider_factory=failing_provider_factory,
        store_factory=lambda **_: FakeVectorStore(),
        retriever_factory=lambda **_: FakeRetriever(
            chunks_by_query={}
        ),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "error: provider setup failed" in captured.err


def test_missing_collection_returns_exit_two_without_creation(
    tmp_path: Path,
    capsys,
):
    cases_path = tmp_path / "cases.jsonl"
    write_cases(
        cases_path,
        [
            RetrievalEvaluationCase(
                case_id="missing-collection-case",
                query="TCP",
                relevant_chunk_ids=["chunk-tcp"],
            )
        ],
    )
    client = QdrantClient(":memory:")
    collection_name = "missing-evaluate-dense"
    provider = LocalModeEmbeddingProvider()
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        dimensions=2,
    )

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--dataset-id",
            "missing-collection-dataset",
            "--collection",
            collection_name,
            "--dimensions",
            "2",
        ],
        provider_factory=lambda **_: provider,
        store_factory=lambda **_: store,
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "does not exist" in captured.err
    assert client.collection_exists(collection_name) is False
