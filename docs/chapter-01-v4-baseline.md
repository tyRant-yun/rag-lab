# Chapter 01 V4 正式基线

**状态**：正式（Official）

本文档把 `computer-networking/output/chapter-01/baseline-v4-final` 定为
计算机网络教材第一章（PDF 第 19–61 页）的正式规范化/切块语料，并记录
审计、评估集与检索基线结果。

## 产物来源

| 项 | 值 |
| --- | --- |
| 源文档 | 《计算机网络：自顶向下方法》第 8 版 第 1 章 |
| 页码范围 | 19–61（43 页） |
| Docling JSON | `chapter-01-pages-019-061.docling.json`（schema 1.10.0） |
| 规范化版本 | `1.1.0` |
| 切块版本 | `1.1.0` |
| 文档 ID | `sha256:fef1b562abea48b141fabdee49cb9bcc7c7fecd400b416affd5811543da52592` |
| 生成日期 | 2026-08-05 |

## 质量快照

### Normalizer（blocks.jsonl，359 块）

| 指标 | 值 |
| --- | ---: |
| 原始块 / 规范化块 | 360 / 359 |
| 移除页眉页脚（furniture） | 65 |
| 阅读顺序重排 | 120 |
| 标题降级 | 37 |
| 短碎片占比 | 4.72% |
| 待复核页 | 20 / 43 |

块类型分布：288 段落 / 28 图注 / 27 小节标题 / 15 列表项 / 1 文档标题；
无空文本；21 块带 `image_path`。

### Chunker（chunks.jsonl，69 块）

| 指标 | 值 |
| --- | ---: |
| 输入块 / 输出块 | 359 / 69 |
| 跨页连接 | 28 |
| 超长正文切分 | 0 |
| 超长原子块 | 0 |
| 带 overlap 的块 | 40（3401 字符） |
| 页码覆盖 | 19–61 全部 43 页 |

正文长度 87–1159 字符（均值 812），`index_text` 最大 1196 ≤ 1200；
`chunk_id` / `content_hash` 零重复。

## 整理决定

1. `baseline-v4-final` 为第一章正式语料；`chapter-01-full` 空目录已移除。
2. 标题/复核审计见 [chapter-01-v4-title-audit.md](./chapter-01-v4-title-audit.md)；
   审计结论与需要进入 Normalizer 规则的条目以该文档为准。
3. 全章评估集为 `evaluations/computer_networking/chapter_01_v4.jsonl`
   （由 `scripts/build_chapter01_eval_cases.py` 生成，
   27 个小节标题探针 + 6 条 smoke 查询合并，全部相关 ID 经语料校验）。

评估集共 33 条：27 条标题探针（query = 小节标题，相关块 =
`heading_path` 包含该标题的全部 Chunk）+ 6 条 smoke 问句式查询
（`protocol-definition` 因相关块 ID 不在 v4 语料中被自动跳过）。

## 检索基线

以下基线均以 `chapter_01_v4.jsonl` 为评估集运行，结果保存在
`evaluations/computer_networking/baselines/`。

### BM25

| Top K | Hit@K | Mean Recall@K | MRR |
| --- | ---: | ---: | ---: |
| 3 | 0.9697 | 0.7206 | 0.9343 |
| 5 | 0.9697 | 0.8642 | 0.9343 |

### Dense（qwen3-embedding:0.6b，1024 维，collection `computer-networking-chapter-01-v4`）

| Top K | Hit@K | Mean Recall@K | MRR |
| --- | ---: | ---: | ---: |
| 1 | 0.8485 | 0.3718 | 0.8485 |
| 3 | 0.8788 | 0.6519 | 0.8636 |
| 5 | 0.9697 | 0.7925 | 0.8833 |

### Embedding 与索引验证

- `embed-chunks`：69 Chunk / 9 batch / 69 向量，1024 维，
  向量模长 0.9999994–1.0000007（有效）。
  embedding_version = `ollama:qwen3-embedding:0.6b:dimensions-1024:query-v1-ec1f1563040d`
- `index-qdrant`：collection `computer-networking-chapter-01-v4`
  已创建并写入 69/69 条。

### Hybrid RRF（BM25 + Dense，rrf_k=60，per_retriever_k=10）

| Top K | Hit@K | Mean Recall@K | MRR |
| --- | ---: | ---: | ---: |
| 3 | 1.0000 | 0.7711 | 0.9394 |
| 5 | 1.0000 | 0.8915 | 0.9343 |

实现见 `src/rag_lab/retrieval/hybrid/` 与
`src/rag_lab/evaluation/hybrid_cli.py`；报告保存在
`evaluations/computer_networking/baselines/chapter_01_v4_hybrid_rrf_top{3,5}.json`。

### Rerank（词法重叠重排，fetch_k=20，权重 1.0/1.0/1.0）

| Top K | Hit@K | Mean Recall@K | MRR |
| --- | ---: | ---: | ---: |
| 3 | 1.0000 | 0.7988 | 0.9848 |
| 5 | 1.0000 | 0.9097 | 0.9848 |

实现见 `src/rag_lab/retrieval/rerank/` 与
`src/rag_lab/evaluation/rerank_cli.py`；报告保存在
`evaluations/computer_networking/baselines/chapter_01_v4_rerank_top{3,5}.json`。
重排后 33 条评估中 32 条首个相关结果排第 1，MRR 从 Hybrid 的 0.9343
提升到 0.9848。

### 失败用例分析（Top 5）

BM25 与 Dense 在 Hit@5 均为 0.9697（32/33），且**未命中用例不同**：

| 用例 | query | BM25 | Dense | Hybrid RRF | Rerank |
| --- | --- | --- | --- | --- | --- |
| `end-systems` | 什么是端系统 | 未命中 | 第 4 名命中 | 命中 | 第 2 名命中 |
| `packet-switching` | 什么是分组交换网络 | 第 3 名命中 | 未命中 | 命中 | 第 1 名命中 |

说明词法检索和向量检索对问句式查询的失败面不重叠，
RRF 混合检索已把 Hit@5 提升到 1.000000（33/33）；叠加词法重排后
MRR 进一步从 0.9343 提升到 0.9848（32/33 排第 1）。

## 复现命令

```powershell
# BM25 评估
evaluate-bm25 `
  --chunks "D:\rag-lab\computer-networking\output\chapter-01\baseline-v4-final\chunked-max1200-overlap120\chunks.jsonl" `
  --cases ".\evaluations\computer_networking\chapter_01_v4.jsonl" `
  --dataset-id "chapter-01-v4" `
  --top-k 5 --json

# Embedding 验证（不写向量）
embed-chunks `
  --chunks "D:\rag-lab\computer-networking\output\chapter-01\baseline-v4-final\chunked-max1200-overlap120\chunks.jsonl" `
  --model "qwen3-embedding:0.6b" --dimensions 1024 --batch-size 8 --json

# 索引全章 collection
index-qdrant `
  --chunks "D:\rag-lab\computer-networking\output\chapter-01\baseline-v4-final\chunked-max1200-overlap120\chunks.jsonl" `
  --collection "computer-networking-chapter-01-v4" `
  --model "qwen3-embedding:0.6b" --dimensions 1024

# Dense 评估
evaluate-dense `
  --cases ".\evaluations\computer_networking\chapter_01_v4.jsonl" `
  --dataset-id "chapter-01-v4" `
  --collection "computer-networking-chapter-01-v4" `
  --top-k 5 --json

# 混合评估
evaluate-hybrid `
  --chunks "path\to\chunks.jsonl" `
  --cases ".\evaluations\computer_networking\chapter_01_v4.jsonl" `
  --dataset-id "chapter-01-v4" `
  --collection "computer-networking-chapter-01-v4" `
  --top-k 5 --json
```

# 重排评估
evaluate-rerank `
  --chunks "path\to\chunks.jsonl" `
  --cases ".\evaluations\computer_networking\chapter_01_v4.jsonl" `
  --dataset-id "chapter-01-v4" `
  --collection "computer-networking-chapter-01-v4" `
  --top-k 5 --json
```

## 已知限制

- 37 个标题被 Normalizer 降级为段落，`heading_path` 可能缺失部分小节，
  影响按小节检索；修正清单见审计文档。
- 4 张无图注图片（`image_000005/000010/000016/000017`）按既有设计未进入
  normalized 产物，图片检索暂不完整。
- `source_path` 为绝对路径，仅用于溯源；产物重新生成后该字段会变化。
