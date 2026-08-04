from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rag_lab.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    KnowledgeChunk,
)
from rag_lab.embeddings.cli import (
    main,
    run_embedding_check,
)


def make_chunk(
    *,
    chunk_id: str,
    index_text: str,
    ordinal: int,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="document-a",
        content=index_text,
        index_text=index_text,
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


def make_chunks() -> tuple[KnowledgeChunk, ...]:
    return (
        make_chunk(
            chunk_id="chunk-1",
            index_text="第一段索引文本",
            ordinal=1,
        ),
        make_chunk(
            chunk_id="chunk-2",
            index_text="第二段索引文本",
            ordinal=2,
        ),
        make_chunk(
            chunk_id="chunk-3",
            index_text="第三段索引文本",
            ordinal=3,
        ),
    )


def write_chunks(
    path: Path,
    chunks: Sequence[KnowledgeChunk],
) -> None:
    content = "\n".join(
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
        )
        for chunk in chunks
    )

    if chunks:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

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
        active_texts = list(texts)
        self.document_calls.append(active_texts)

        vectors = [
            EmbeddingVector(
                values=[0.6, 0.8],
                dimensions=2,
            )
            for _ in active_texts
        ]

        return EmbeddingBatch(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            vectors=vectors,
            input_count=len(active_texts),
            elapsed_ms=1.0,
            embedding_version=(
                self.embedding_version
            ),
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            values=[0.6, 0.8],
            dimensions=2,
        )


class FakeProviderFactory:
    def __init__(
        self,
        provider: FakeEmbeddingProvider,
    ) -> None:
        self.provider = provider
        self.arguments: dict[str, object] | None = None

    def __call__(
        self,
        **arguments: object,
    ) -> FakeEmbeddingProvider:
        self.arguments = arguments
        return self.provider


def test_embedding_check_batches_index_text():
    provider = FakeEmbeddingProvider()

    report = run_embedding_check(
        chunks=make_chunks(),
        provider=provider,
        batch_size=2,
    )

    assert provider.document_calls == [
        [
            "第一段索引文本",
            "第二段索引文本",
        ],
        [
            "第三段索引文本",
        ],
    ]

    assert report.chunk_count == 3
    assert report.batch_count == 2
    assert report.vector_count == 3
    assert report.minimum_vector_norm == 1.0
    assert report.maximum_vector_norm == 1.0


def test_cli_writes_json_report(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, make_chunks())

    provider = FakeEmbeddingProvider()
    factory = FakeProviderFactory(provider)

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--model",
            "requested-model",
            "--dimensions",
            "2",
            "--batch-size",
            "2",
            "--json",
        ],
        provider_factory=factory,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["chunk_count"] == 3
    assert payload["batch_count"] == 2
    assert payload["vector_count"] == 3

    assert factory.arguments is not None
    assert factory.arguments["model_name"] == (
        "requested-model"
    )
    assert factory.arguments["dimensions"] == 2


def test_cli_writes_human_report(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, make_chunks())

    exit_code = main(
        [
            "--chunks",
            str(path),
        ],
        provider_factory=FakeProviderFactory(
            FakeEmbeddingProvider()
        ),
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Provider: fake" in captured.out
    assert "Chunks: 3" in captured.out
    assert "Batches: 1" in captured.out
    assert "Vectors: 3" in captured.out


def test_cli_reports_missing_file(
    tmp_path: Path,
    capsys,
):
    exit_code = main(
        [
            "--chunks",
            str(tmp_path / "missing.jsonl"),
        ],
        provider_factory=FakeProviderFactory(
            FakeEmbeddingProvider()
        ),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_cli_reports_empty_chunks(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--chunks",
            str(path),
        ],
        provider_factory=FakeProviderFactory(
            FakeEmbeddingProvider()
        ),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "chunks cannot be empty" in captured.err


def test_cli_rejects_invalid_batch_size(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, make_chunks())

    exit_code = main(
        [
            "--chunks",
            str(path),
            "--batch-size",
            "0",
        ],
        provider_factory=FakeProviderFactory(
            FakeEmbeddingProvider()
        ),
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert (
        "batch_size must be at least 1"
        in captured.err
    )
