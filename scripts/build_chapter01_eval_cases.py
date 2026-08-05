"""Build the chapter-01 v4 retrieval evaluation case set.

Each section heading in the normalized baseline becomes one probe: the query
is the heading text and the relevant chunks are every chunk whose
heading_path contains that exact heading.  Existing smoke cases are merged
when their relevant chunk ids exist in the target corpus.
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

    cases: list[dict] = []
    for block in headings:
        heading = (block.get("text") or "").strip()
        if not heading:
            continue
        relevant = [
            chunk["chunk_id"]
            for chunk in sorted(
                chunks, key=lambda item: item.get("ordinal", 0)
            )
            if heading in chunk.get("heading_path", [])
        ]
        case_id = (
            f"section-{block.get('ordinal', 0):03d}-"
            f"{_slug(heading)}"
        )
        cases.append(
            {
                "case_id": case_id,
                "query": heading,
                "relevant_chunk_ids": relevant,
                "filters": None,
            }
        )

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
        f"section cases: {len(headings)}  "
        f"smoke merged: {merged_smoke}  total: {len(cases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
