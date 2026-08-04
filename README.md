# RAG Lab

RAG Lab 是一个本地知识库项目，用于把源文档转换成可追踪、可复现的数据，供后续检索、RAG 和 Agent 使用。

当前已经实现：

1. Docling 文档规范化；
2. 结构感知的知识切块；
3. 确定性的 JSONL、Markdown 和质量报告输出；
4. 中文词法分析；
5. 内存 BM25 索引、过滤和检索；
6. 可复用的检索评估模型、评估器和 BM25 CLI。
7. Ollama 安装及 ollama pull；
8. embed-chunks 示例；
9. 8 个 Chunk、2 个 batch、1024 维；
10. 向量模长范围；

明确报告不包含原始向量；
下一阶段是 Qdrant。


## 整体流程

```text
源 PDF
  ↓
Docling JSON
  ↓ Normalizer
NormalizedBlock / blocks.jsonl
  ↓ Chunker
KnowledgeChunk / chunks.jsonl
  ↓ LexicalAnalyzer
中文词项
  ↓ BM25Index
内存词法索引
  ↓ BM25Retriever
SearchResult
  ↓ RetrievalEvaluator
BM25 基线报告
  ↓ 后续阶段
Embedding → Qdrant → 混合检索 → RAG → Agent
```

Markdown 文件只用于人工检查，JSONL 文件才是下游机器接口。

## 模块职责

- **Docling 转换层**：负责 PDF 解析、OCR、版面识别、表格、图片以及 Docling JSON 输出。
- **Normalizer**：负责阅读顺序、文本清理、块类型、标题路径、页码来源和图片引用。
- **Chunker**：负责跨页段落连接、语义边界、长度控制、来源聚合和稳定 Chunk ID。
- **LexicalAnalyzer**：负责 Unicode 规范化、中文分词、技术词、停用词和领域词处理。
- **BM25Index/Retriever**：负责内存词法索引、BM25 评分、元数据过滤和稳定排名。
- **Evaluation**：负责标注查询读取、Hit@K、Recall@K、MRR 和不同 Retriever 的统一比较。
- **Dense/Hybrid Retriever**：后续负责 Embedding、Qdrant 和混合排序。
- **Agent**：后续通过稳定的检索工具使用知识库。

Chunker 只读取 `blocks.jsonl`，不读取 PDF、Docling JSON 或人工审阅 Markdown。

## Python 目录结构

```text
src/rag_lab/
├── contracts/
│   ├── blocks.py
│   ├── chunks.py
│   └── search.py
├── normalization/
│   ├── cli.py
│   ├── models.py
│   ├── normalizer.py
│   └── serialization.py
├── chunking/
│   ├── cli.py
│   ├── models.py
│   ├── chunker.py
│   └── serialization.py
├── retrieval/
│   ├── serialization.py
│   ├── lexical/
│   │   └── analyzer.py
│   └── bm25/
│       ├── index.py
│       ├── retriever.py
│       └── cli.py
└── evaluation/
    ├── models.py
    ├── serialization.py
    ├── evaluator.py
    └── bm25_cli.py
```

`rag_lab.contracts` 存放共享产品契约，导入它不会加载 Normalizer、
Chunker 或 Retriever 的具体实现。

## NormalizedBlock 契约

`blocks.jsonl` 的每一行都是一个经过严格校验的 `NormalizedBlock`。

主要字段：

- `block_id`：规范化 Block 的确定性 SHA-256 ID，不依赖输出顺序；
- `document_id`：根据源 PDF 内容生成的文档 ID；
- `text`：规范化后的文本；
- `block_type`：段落、标题、列表、表格等结构类型；
- `heading_path`：当前 Block 所属的完整标题路径；
- `page_start`、`page_end`：源 PDF 页码范围；
- `ordinal`：从 1 开始的文档阅读顺序；
- `source_path`：源文件路径；
- `image_path`：可选图片路径，相对于 Normalizer 产物目录；
- `normalization_version`：Normalizer 规则版本。

Normalizer 输出：

- `blocks.jsonl`：下游结构化接口；
- `document.md`：人工审阅版本；
- `normalization-report.json`：规范化质量报告。

`block_id` 使用文档 ID、Docling 源引用、页码、内容结构和
Normalizer 版本计算，不包含 `ordinal`。因此在前方插入新 Block
时，后方来源与内容未变化的 Block ID 保持不变。

对于带图片的 Docling 文档，Normalizer 通过
`pictures[*].captions[*].$ref` 将图注关联到图片，把资源复制到
Normalizer 产物目录，并在 JSONL 和审阅 Markdown 中保存相对路径。
直接调用 `write_normalization_outputs()` 时，如果没有传入
`asset_source_directory`，所有引用图片必须已经存在于输出目录，
否则函数会拒绝生成不完整的文档 bundle。

## KnowledgeChunk 契约

`chunks.jsonl` 的每一行都是一个经过严格校验的 `KnowledgeChunk`。

主要字段：

- `chunk_id`：稳定且可复现的 Chunk ID；
- `document_id`：源文档 ID；
- `content`：用于回答和引用的正文；
- `index_text`：用于词法检索和 Embedding 的文本；
- `heading_path`：Chunk 所属标题路径；
- `page_start`、`page_end`：来源页码范围；
- `ordinal`：Chunk 在当前文档中的显示顺序；
- `block_ids`：组成 Chunk 的有序且去重的 Block ID；
- `content_hash`：正文内容哈希；
- `normalization_version`：Normalizer 版本；
- `chunking_version`：Chunker 版本。

`chunk_id` 故意不包含输出 `ordinal`。因此，在文档前面新增 Chunk 时，后面未变化的 Chunk ID 不会全部改变。

## Chunker 当前规则

当前 Chunker 会：

1. 验证一次输入只包含一个文档；
2. 按源 Block 的 `ordinal` 排序；
3. 将文档标题和章节标题作为控制块；
4. 按 `heading_path` 对连续正文块分组；
5. 保守识别并连接跨页段落；
6. 把标题上下文计入长度限制；
7. 使用贪心算法将正文装入 Chunk；
8. 优先在完整句子边界切分超长正文；
9. 对仍然过长的单句使用字符级保底切分；
10. 保持 table、code、equation 的原子性；
11. 保留源 Block ID 和页码来源；
12. 在相同 `heading_path` 内添加 best-effort overlap；
13. 生成确定性的内容哈希和 Chunk ID。

默认 overlap 为 120 个字符。Chunker 优先重复完整内容单元；
完整单元过大时，只对段落、列表和图注使用完整句子后缀。
table、code、equation 不会为了 overlap 被拆开。若下一个 Chunk
没有足够空间，实际 overlap 可以小于配置值，甚至为 0。

## 开发环境

创建虚拟环境：

```powershell
py -3.12 -m venv .venv

& ".\.venv\Scripts\Activate.ps1"
```

安装项目和测试工具：

```powershell
python -m pip install -e .
python -m pip install pytest
```

运行测试：

```powershell
python -m pytest -q
```

## 运行 Normalizer

使用 Python 模块入口：

```powershell
python -m rag_lab.normalization.cli `
  --input-json "path\to\document.docling.json" `
  --source "path\to\source.pdf" `
  --output "path\to\normalized" `
  --normalization-version "1.1.0"
```

使用安装后的命令：

```powershell
normalize-docling `
  --input-json "path\to\document.docling.json" `
  --source "path\to\source.pdf" `
  --output "path\to\normalized" `
  --normalization-version "1.1.0"
```

Normalizer 产物：

```text
normalized/
├── blocks.jsonl
├── document.md
└── normalization-report.json
```

## 运行 Chunker

使用 Python 模块入口：

```powershell
python -m rag_lab.chunking.cli `
  --input "path\to\normalized\blocks.jsonl" `
  --output "path\to\chunked" `
  --max-chars 1200 `
  --overlap-chars 120 `
  --chunking-version "1.1.0"
```

使用安装后的命令：

```powershell
chunk-normalized `
  --input "path\to\normalized\blocks.jsonl" `
  --output "path\to\chunked" `
  --max-chars 1200 `
  --overlap-chars 120 `
  --chunking-version "1.1.0"
```

Chunker 产物：

```text
chunked/
├── chunks.jsonl
├── chunks.md
└── chunking-report.json
```

其中：

- `chunks.jsonl`：后续索引和检索使用的结构化接口；
- `chunks.md`：人工检查 Chunk 内容和来源；
- `chunking-report.json`：Chunker 处理统计。

## 运行 BM25 检索

以下命令中的 `path\to\...` 是占位符，必须替换成实际文件路径。

```powershell
search-bm25 `
  --chunks "path\to\chunks.jsonl" `
  --query "什么是协议" `
  --top-k 5
```

使用 `--json` 输出完整 `SearchResult`。可以重复传入
`--document-id`、`--heading`、`--user-word` 和 `--stopword`，
并使用 `--page-start`、`--page-end` 过滤来源页码。

当前 BM25 索引只驻留内存，每次 CLI 调用都会重新读取 Chunk、
执行词法分析并构建索引。

## 运行 BM25 评估

以下命令中的 `path\to\...` 同样是占位符，必须替换成实际文件路径。

```powershell
evaluate-bm25 `
  --chunks "path\to\chunks.jsonl" `
  --cases "path\to\evaluation-cases.jsonl" `
  --dataset-id "chapter-01-smoke" `
  --top-k 5
```

在当前 worktree 根目录下，可以直接运行本地 smoke 基线：

```powershell
evaluate-bm25 `
  --chunks "D:\rag-lab\computer-networking\output\chapter-01-smoke\baseline-v1.1\chunked-max1200-overlap120\chunks.jsonl" `
  --cases ".\evaluations\computer_networking\chapter_01_smoke.jsonl" `
  --dataset-id "chapter-01-smoke" `
  --top-k 5
```

使用 `--json` 输出完整 `RetrievalEvaluationReport`。评估器计算
Hit@K、Mean Recall@K 和 MRR，并验证标注的相关 Chunk ID 是否
存在于当前语料。

## 质量报告

接受 Normalizer 结果前，应检查 `normalization-report.json`，尤其是：

```text
pages_requiring_review
downgraded_heading_count
removed_furniture_count
reordered_block_count
```

接受 Chunker 结果前，应检查 `chunking-report.json`：

```text
input_block_count
output_chunk_count
cross_page_join_count
long_block_split_count
oversized_atomic_block_count
overlapped_chunk_count
overlap_char_count
```

计数不为零不一定表示错误，而是表示应该在 `chunks.md` 中重点检查相关内容。

## 当前限制

- Normalizer 阅读顺序恢复目前面向单栏教材。
- 多章节教材依赖章节开页中独立的 `第 N 章` 标记；普通页眉中的
  章节名称不会触发章节切换。
- 只有通过 Docling caption 引用关联的图片会进入 `image_path`；
  无图注图片目前不会生成独立 Block。
- OCR 和 Docling 本身的识别错误不会被 Chunker修复。
- 标题质量依赖 Normalizer 的标题分类。
- 跨页连接使用页码、顺序和标点启发式规则，无法覆盖所有复杂版面。
- `max_chars` 计算 Unicode 字符数，而不是模型 Token 数。
- table、code、equation 即使超长也保持原子性。
- overlap 是不超过目标字符数的 best-effort 结果，不保证每个
  Chunk 都达到目标值。
- overlap 不跨 `heading_path`，也不会截断 table、code、equation。
- Normalizer 和 Chunker 的一次 CLI 调用只处理一个文档。
- BM25 索引当前只驻留内存，每次进程启动都会重新构建。
- 尚未实现增量索引、删除、持久化词法索引、Embedding、Qdrant、
  混合检索、检索 API 和 Agent。
- 当前评估集只有第 19–23 页的 8 个 Chunk 和 7 个查询，只能作为
  流程冒烟基线，不能代表完整第一章的检索质量。

## 真实样本回归

使用一本计算机网络教材的 PDF 第 19–23 页进行本地回归：

```text
原始 Block：31
NormalizedBlock：30
阅读顺序调整：8
KnowledgeChunk：8
跨页段落连接：3
超长正文切分：0
超长原子块：0
带 overlap 的 Chunk：3
overlap 字符总数：266
```

回归结果：

- 25 个正文 Block 全部进入最终 Chunk；
- Chunk ID 全部唯一；
- Chunk ordinal 连续；
- 页码全部保持在 19–23；
- 没有空正文；
- 没有遗漏或意外加入控制块；
- 所有 `index_text` 均未超过 1200 字符；
- 3 个 Chunk 获得 72、93、101 字符的句子级 overlap；
- overlap 均位于相同 `heading_path`，没有跨章节传播；
- 已知的“用于存／储和传输”跨页断句被正确恢复；
- 内存重新计算结果与磁盘 JSONL 完全一致。

第 23 页存在一个上游审阅项：

```text
“1. 人类活动的类比”
```

该文本被 Normalizer 保守地降级为普通段落，没有进入 `heading_path`。内容没有丢失，也不影响当前 Chunker 验收，但会轻微降低该小节的标题检索效果。

本地 PDF、Docling 产物、NormalizedBlock 和 Chunk 产物均不提交到 Git。

### BM25 冒烟基线

使用相同的 8 个 Chunk 和 7 个标注查询，以 `top_k=5` 运行：

```text
Hit@5：1.000000
Mean Recall@5：1.000000
MRR：0.857143
```

其中 `internet-definition` 和 `packet-switching` 的首个相关结果
位于第 2 名，其余 5 个查询的首个相关结果位于第 1 名。

基线报告保存在：

```text
evaluations/computer_networking/baselines/
chapter_01_smoke_bm25_top5.json
```

该结果只用于验证评估链路和记录词法检索起点，不应解读为完整
知识库上的质量结论。

## 后续计划

已经完成：

1. 中文词法处理；
2. BM25 检索与基线评估。

下一阶段依次实现：

1. Ollama Embedding；
2. Qdrant 向量存储；
3. BM25 与向量混合检索；
4. 检索 API；
5. Agent 工具接入。
