"""Audit a chapter baseline bundle and emit a title-review Markdown report.

Read-only analysis of a baseline directory produced by the normalizer and
chunker.  The report covers:

- pages that require review (from normalization-report.json);
- heading-like blocks that were classified as paragraph or list items
  (candidates for downgraded headings);
- figure captions and images missing from the normalized bundle; and
- chunk page coverage.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys


HEADING_LIKE_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+){1,3}\s|第[一二三四五六七八九十百千0-9]+[章节篇部分])"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _heading_candidates(blocks: list[dict]) -> list[dict]:
    candidates = []
    for block in blocks:
        if block.get("block_type") not in ("paragraph", "list_item"):
            continue
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if HEADING_LIKE_PATTERN.match(text):
            candidates.append(block)
            continue
        if len(text) <= 24 and not text.rstrip().endswith(
            ("。", "；", "，", "！", "？", ".", ";", ",")
        ):
            candidates.append(block)
    return candidates


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    baseline = arguments.baseline.resolve()
    report_path = baseline / "normalized" / "normalization-report.json"
    blocks_path = baseline / "normalized" / "blocks.jsonl"
    chunks_path = (
        baseline / "chunked-max1200-overlap120" / "chunks.jsonl"
    )

    if not report_path.is_file() or not blocks_path.is_file():
        print(
            f"missing normalized artifacts under {baseline}",
            file=sys.stderr,
        )
        return 2

    report = _read_json(report_path)
    blocks = _read_jsonl(blocks_path)

    heading_blocks = [
        block
        for block in blocks
        if block.get("block_type")
        in ("document_title", "section_heading")
    ]
    candidates = _heading_candidates(blocks)
    captions = [
        block
        for block in blocks
        if block.get("block_type") == "figure_caption"
    ]

    review_pages = list(report.get("pages_requiring_review", []))
    page_context: dict[int, str] = {}
    for block in blocks:
        page = block.get("page_start")
        if page in review_pages and page not in page_context:
            page_context[page] = " / ".join(
                block.get("heading_path", [])
            )

    missing_images: list[str] = []
    root_assets = baseline / "assets"
    normalized_assets = baseline / "normalized" / "assets"
    if root_assets.is_dir() and normalized_assets.is_dir():
        root_names = {path.name for path in root_assets.iterdir()}
        normalized_names = {
            path.name for path in normalized_assets.iterdir()
        }
        missing_images = sorted(root_names - normalized_names)

    lines: list[str] = []
    lines.append(f"# Chapter 01 Baseline Audit — {baseline.name}")
    lines.append("")
    lines.append(
        f"- 源文档: `{report.get('source_path', 'unknown')}`"
    )
    lines.append(
        f"- 文档 ID: `{report.get('document_id', 'unknown')}`"
    )
    lines.append(
        f"- 规范化版本: "
        f"{report.get('normalization_version', 'unknown')}"
    )
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    lines.append(f"- normalized blocks: {len(blocks)}")
    lines.append(f"- heading blocks: {len(heading_blocks)}")
    lines.append(f"- figure captions: {len(captions)}")
    lines.append(
        "- 降级标题候选"
        f"（段落/列表项中像标题的文本）: {len(candidates)}"
    )
    lines.append(f"- 待复核页: {len(review_pages)}")
    lines.append("")

    lines.append("## 待复核页上下文")
    lines.append("")
    if review_pages:
        lines.append("| 页码 | 该页首个 heading_path |")
        lines.append("| ---: | --- |")
        for page in review_pages:
            lines.append(
                f"| {page} | {page_context.get(page, '—')} |"
            )
    else:
        lines.append("无。")
    lines.append("")

    lines.append("## 降级标题候选")
    lines.append("")
    lines.append(
        "以下文本被分类为段落或列表项，但形态上像标题。"
        "需要对照 `document.md` 人工确认。"
    )
    lines.append("")
    if candidates:
        lines.append("| ordinal | 页码 | block_type | 文本 |")
        lines.append("| ---: | ---: | --- | --- |")
        for block in candidates:
            text = (block.get("text") or "").replace("|", "\\|")
            lines.append(
                f"| {block.get('ordinal')} | "
                f"{block.get('page_start')} | "
                f"{block.get('block_type')} | {text[:80]} |"
            )
    else:
        lines.append("无候选。")
    lines.append("")

    lines.append("## 图片覆盖")
    lines.append("")
    if missing_images:
        lines.append(
            f"根目录 assets 中有 {len(missing_images)} 张图未进入 "
            "normalized 产物（按设计仅带图注图片入库）："
        )
        lines.append("")
        for name in missing_images:
            lines.append(f"- `{name}`")
    else:
        lines.append("所有图片均已进入 normalized 产物。")
    lines.append("")

    if chunks_path.is_file():
        chunks = _read_jsonl(chunks_path)
        covered: set[int] = set()
        for chunk in chunks:
            covered.update(
                range(
                    chunk.get("page_start", 0),
                    chunk.get("page_end", 0) + 1,
                )
            )
        expected = set(report.get("source_pages", []))
        lines.append("## Chunk 页覆盖")
        lines.append("")
        lines.append(f"- chunks: {len(chunks)}")
        if expected:
            lines.append(
                f"- 覆盖页数: {len(expected & covered)}/{len(expected)}"
            )
            missing_pages = sorted(expected - covered)
            if missing_pages:
                lines.append(f"- 缺失页: {missing_pages}")
        lines.append("")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"audit report written: {arguments.output}")
    print(
        f"review_pages={len(review_pages)} "
        f"heading_candidates={len(candidates)} "
        f"missing_images={len(missing_images)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
