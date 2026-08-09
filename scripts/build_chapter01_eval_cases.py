"""Build reproducible chapter-01 retrieval evaluation case sets.

Each section heading in the normalized baseline becomes one probe: the query
is the heading text and the relevant chunks are every chunk whose
heading_path contains that exact heading.  Existing smoke cases are merged
when their relevant chunk ids exist in the target corpus.  A section template
keeps stable case IDs across artifact versions, while semantic probes derive
new relevant IDs from headings and source phrases in the target corpus.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _slug(text: str) -> str:
    value = re.sub(
        r"[^\w\u4e00-\u9fff]+", "-", text.strip()
    ).strip("-")
    return value or "heading"


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--smoke-cases", type=Path)
    parser.add_argument(
        "--section-template",
        type=Path,
        help=(
            "Existing JSONL whose section case IDs and queries should be "
            "re-bound to this corpus."
        ),
    )
    parser.add_argument(
        "--semantic-probes",
        type=Path,
        help=(
            "JSONL with case_id, query, heading, and optional text fields "
            "used to derive target-corpus relevance labels."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    blocks = _read_jsonl(arguments.blocks)
    chunks = _read_jsonl(arguments.chunks)
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    headings = [
        block
        for block in blocks
        if block.get("block_type") == "section_heading"
    ]
    heading_texts = {
        (block.get("text") or "").strip()
        for block in headings
    }

    if arguments.section_template:
        section_specs = [
            {
                "case_id": case["case_id"],
                "query": case["query"],
            }
            for case in _read_jsonl(arguments.section_template)
            if str(case.get("case_id", "")).startswith("section-")
        ]
    else:
        section_specs = [
            {
                "case_id": (
                    f"section-{block.get('ordinal', 0):03d}-"
                    f"{_slug((block.get('text') or '').strip())}"
                ),
                "query": (block.get("text") or "").strip(),
            }
            for block in headings
            if (block.get("text") or "").strip()
        ]

    cases: list[dict] = []
    for spec in section_specs:
        heading = spec["query"]
        if heading not in heading_texts:
            raise SystemExit(
                "section template heading missing from target corpus: "
                f"{heading}"
            )
        relevant = [
            chunk["chunk_id"]
            for chunk in sorted(
                chunks, key=lambda item: item.get("ordinal", 0)
            )
            if heading in chunk.get("heading_path", [])
        ]
        cases.append(
            {
                "case_id": spec["case_id"],
                "query": heading,
                "relevant_chunk_ids": relevant,
                "filters": None,
            }
        )

    semantic_merged = 0
    if arguments.semantic_probes:
        for probe in _read_jsonl(arguments.semantic_probes):
            heading = probe.get("heading")
            if not isinstance(heading, str) or not heading.strip():
                raise SystemExit(
                    "semantic probe requires a non-empty heading"
                )
            text = probe.get("text")
            if text is not None and not isinstance(text, str):
                raise SystemExit(
                    "semantic probe text must be a string when provided"
                )
            relevant = [
                chunk["chunk_id"]
                for chunk in sorted(
                    chunks, key=lambda item: item.get("ordinal", 0)
                )
                if heading in chunk.get("heading_path", [])
                and (text is None or text in chunk.get("content", ""))
            ]
            if not relevant:
                raise SystemExit(
                    "semantic probe found no relevant chunks: "
                    f"{probe.get('case_id')}"
                )
            cases.append(
                {
                    "case_id": probe["case_id"],
                    "query": probe["query"],
                    "relevant_chunk_ids": relevant,
                    "filters": probe.get("filters"),
                }
            )
            semantic_merged += 1

    merged_smoke = 0
    if arguments.smoke_cases and arguments.smoke_cases.is_file():
        for case in _read_jsonl(arguments.smoke_cases):
            requested = case.get("relevant_chunk_ids", [])
            missing = [
                case_id
                for case_id in requested
                if case_id not in chunk_ids
            ]
            if missing:
                print(
                    f"warning: drop smoke case "
                    f"{case.get('case_id')}: {len(missing)} ids "
                    "missing from corpus",
                    file=sys.stderr,
                )
            existing = [
                case_id
                for case_id in requested
                if case_id in chunk_ids
            ]
            if not existing:
                print(
                    f"warning: skip empty smoke case "
                    f"{case.get('case_id')}",
                    file=sys.stderr,
                )
                continue
            cases.append(
                {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "relevant_chunk_ids": existing,
                    "filters": case.get("filters"),
                }
            )
            merged_smoke += 1

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        print("error: duplicate case ids generated", file=sys.stderr)
        return 2

    empty = [
        case["case_id"]
        for case in cases
        if not case["relevant_chunk_ids"]
    ]
    if empty:
        print(
            f"warning: cases with no relevant chunks: {empty}",
            file=sys.stderr,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(case, ensure_ascii=False) + "\n"
            )

    print(f"eval cases written: {arguments.output}")
    print(
        f"section cases: {len(section_specs)}  "
        f"semantic merged: {semantic_merged}  "
        f"smoke merged: {merged_smoke}  total: {len(cases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
