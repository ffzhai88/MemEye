# CLAUDE.md

## 项目定位

这是 MemEye 项目：一个面向“多模态 agent long-term memory”评测与基准测试的 Python 工程。

核心目标是：
- 评估 AI agent 在长期对话/多轮任务中是否真正记住并利用视觉证据；
- 比较不同 memory 方法（全上下文、检索、总结、agentic memory 等）的表现；
- 用 MCQ / Open-ended 两种方式评估模型，并支持 LLM-as-a-Judge 打分。

## 这套代码主要做什么

本工程不是普通的应用程序，而是一个 benchmark / evaluation 框架，主要流程是：
1. 加载任务数据（对话 JSON + 图片）；
2. 用不同方法（method）运行模型；
3. 生成预测结果；
4. 计算指标（EM、F1、BLEU、BERTScore、LLM Judge 等）；
5. 把结果写入 runs/ 目录，方便分析与对比。

## 关键入口文件

- run_benchmark.py
  - 单任务评测主入口。
- run_matrix.py
  - 批量跑多个 method / model 的对比矩阵。
- score_locked_llm_judge.py
  - 对 open-ended 结果做后处理和 LLM Judge 打分。
- register_external_data.py
  - 从外部数据目录生成 task config。

## 代码结构层次（建议优先读的顺序）

这个仓库的主脉络并不是一个普通 Web 服务，而是一个“配置驱动的评测流水线”。你后续做研究时，最重要的不是从 README 一路往下看，而是先理解这几个层次：

### 1. 顶层入口：怎么启动一次实验

- run_benchmark.py
  - 单任务实验入口。
  - 负责解析命令行参数（task / model / method / output / mode / metric 开关）。
  - 最终调用 benchmark.run_modular_benchmark()。
- run_matrix.py
  - 批量跑多个 model × method 的对照实验。
  - 适合做实验矩阵、对比不同 memory method 的效果。
- score_locked_llm_judge.py
  - 对 open-ended 结果做后处理，并调用 LLM Judge 进行打分。
  - 适合修改评估指标或 Judge prompt 的时候看。
- register_external_data.py
  - 从本地 data/ 目录生成 task configs，供实验使用。

### 2. benchmark/：真正的执行核心

这一层是你改代码最频繁的地方。

- benchmark/common.py
  - YAML / JSON 读取、路径解析、结果写入工具函数。
  - 如果你要改“配置加载”“结果保存”“路径解析”这类基础逻辑，先看这里。
- benchmark/dataset.py
  - 数据加载器，负责把原始对话 JSON + 图片路径解析成可执行的任务对象。
  - 主要功能包括：
    - 解析 multi_session_dialogues；
    - 解析 QA；
    - 构造 history；
    - 解析 question image / image path。
  - 如果你要改数据格式、图片解析、历史上下文构造，先看这里。
- benchmark/methods.py
  - memory method 的注册中心与抽象接口。
  - 这里定义了各种方法的历史上下文构建逻辑，例如：
    - full_context_*：全历史上下文；
    - semantic_rag_*：检索式上下文；
    - clue_only_context：只用 clue 对应轮次；
    - m2a / mma / simplemem / memoryos / reflexion 等外部方法的入口。
  - 如果你的实验涉及“换一种 memory 策略”，基本都从这层开始。
- benchmark/retrieval.py
  - 检索式方法的核心实现。
  - 负责：
    - 用 question 检索相关 round；
    - 生成候选上下文；
    - 支持 sparse / dense / multimodal embedding 检索。
  - 如果你要做检索增强、候选召回、top-k、neighbor window、embedding 模型，优先看这里。
- benchmark/embeddings.py
  - 文本与多模态 embedding 的包装器。
  - 这里是 text embedder / multimodal embedder 的实现。
  - 如果你要替换 embedding 模型、调试向量召回质量，重点看这里。
- benchmark/evaluator.py
  - 评分逻辑中心。
  - 包含：
    - MCQ 选项抽取；
    - open-ended 的 F1 / BLEU / BERTScore；
    - LLM-as-a-Judge 的解析与打分；
    - 汇总统计。
  - 如果你要改评测指标、打分规则、MCQ 解码逻辑，优先看这个文件。
- benchmark/runner.py
  - 真正的执行编排器。
  - 负责读取 task / model / method 配置，实例化 router，构造 dataset，调用 evaluator，写结果到 runs/。
  - 这是从“配置”到“实验结果”之间的总控中心。
- benchmark/matrix.py
  - 多模型、多方法的矩阵总结器。
  - 负责生成 summary.csv / summary.md / summary.json。

### 3. 执行流水线：从配置到结果的完整链路

理解这个调度路径对研究 memory method 至关重要。核心流程如下：

```
run_modular_benchmark() / run_benchmark_matrix()
         │
         ▼
    runner.py :: run_benchmark()
         │
         ├── 1. 加载 YAML 配置（task + model + method）
         ├── 2. 解析数据集，实例化 MemoryBenchmarkDataset
         ├── 3. 调用 methods.py :: get_method(name, config)
         │       返回 HistoryMethod 子类实例
         │
         ├── 4. 判断：是 Agentic 方法（有 .answer()）？
         │
         ├── [非 Agentic 路径]  → baseline / RAG 类方法
         │    ├── method.build_history(dataset, qa)
         │    │    └── 检索类 → retrieval.select_round_ids_for_qa()
         │    │         └── embeddings.py 向量检索
         │    ├── router.answer(history, question)  → 生成回答
         │    └── evaluator 计算指标
         │
         └── [Agentic 路径]  → M2A / MMA 等自主方法
              ├── method.answer(dataset, qa, question)
              ├── 内部自行管理记忆构建与 LLM 调用
              └── evaluator 计算指标
```

结果写入 runs/\<run_dir\>/（config.json, metrics.json, predictions.jsonl）。

理解后可以快速定位研究改动落在哪一层：
- 改”记忆构建策略” → methods.py / retrieval.py
- 改”模型调用方式” → router/ 或 agentic method 内部
- 改”评分方式” → evaluator.py

### 4. router/：模型调用入口

- router/openai_api.py — OpenAI 兼容 API（GPT-4o, GPT-4.1, o3 等）
- router/gemini_api.py — Google Gemini API（Gemini 2.0/2.5 系列）
- router/qwen_local.py — 本地 Qwen 模型加载与推理

### 5. config/methods/：memory method 配置入口

config/methods/ 下有 40+ 个 YAML，每个 method 对应一个或多个配置。典型结构：

```yaml
# 以 semantic_rag_multimodal.yaml 为例
method: semantic_rag_multimodal    # 匹配 methods.py 注册名
retrieval:
  mode: dense_multimodal
  top_k: 5
  neighbor_window: 2
context_token_limit: 128000
```

研究 method 时，大部分实验修改可以先从 YAML 开始，再落到 Python 实现。

### 6. 所有 memory method 详解

所有方法按架构分为三类。这是你研究 method 最核心的参考。

#### 非 Agentic 基线（Non-Agentic Baselines）

通过 `build_history()` 构造历史上下文，交由 `router.answer()` 统一推理。

| 方法名 | 配置文件 | 核心行为 |
|---|---|---|
| `full_context_text_only` | `full_context_text_only.yaml` | 全量历史 + 所有 session，图片替换为 caption，超限截断最早轮次。**FUMemory 文本基线**。 |
| `full_context_multimodal` | `full_context_multimodal.yaml` | 同上，保留图片。**MMFUMemory 多模态基线**。 |
| `full_context_no_visual` | `full_context_no_visual.yaml` | 全文本，无图片无 caption。测试**纯文本泄露量**。 |
| `question_only` | `question_only.yaml` | 空历史。测试 **MCQ guessability**（零上下文消融）。 |
| `target_session_context` | `target_session_context.yaml` | 仅包含 QA 标记的目标 session。 |
| `clue_only_context` | `clue_only_context.yaml` | **Oracle 检索**：仅含 clue 标注的精确轮次。检索上界。 |
| `semantic_rag_text_only` | `semantic_rag_text_only.yaml` | all-MiniLM-L6-v2 编码+余弦检索 top-K。纯文本。 |
| `semantic_rag_multimodal` | `semantic_rag_multimodal.yaml` | 同上，加 SigLIP2/CLIP 图编码，文本+图加权检索。 |

继承关系：
```
HistoryMethod (ABC)
  └── _MemGalleryHistoryMethod
        ├── _MemGalleryFullContextMethod
        │     ├── FullContextTextMethod          (full_context_text_only)
        │     ├── FullContextMultimodalMethod     (full_context_multimodal)
        │     └── FullContextNoVisualMethod       (full_context_no_visual)
        ├── TargetSessionContextMethod           (target_session_context)
        └── ClueOnlyContextMethod                (clue_only_context)
  └── _RetrievalHistoryMethod                    → 委托 retrieval.py
        ├── SemanticRAGTextMethod                (semantic_rag_text_only)
        └── SemanticRAGMultimodalMethod          (semantic_rag_multimodal)
  └── QuestionOnlyMethod                         (question_only)
```

#### Agentic 自主方法

不走 `router.answer()`，实现自己的 `answer()` 控制全部执行逻辑。

**M2A（`m2a`）** — Two-phase pipeline，复现 M2A 论文。
- Phase 1（`process_all_sessions`）：ChatAgent + MemoryManager ReAct 循环消化对话，提取语义记忆存入 SemanticStore。
- Phase 2（`answer_question`）：查询 SemanticStore 生成答案。
- 依赖：TextEmbedder（all-MiniLM-L6-v2）、MultimodalEmbedder（SigLIP2/CLIP）
- 关键文件：`m2a/system.py`（编排器）、`chat_agent.py`、`memory_manager.py`、`stores.py`
- 配置：`m2a.yaml`

**MMA（`mma`）** — Confidence-aware 多模态记忆 agent。
- 核心：置信度加权检索，三个分量 → 来源可信度（0.45）+ 时间衰减（0.40，30天半衰期）+ 冲突共识（0.15）
- 关键文件：`mma/system.py`、`mma/confidence.py`（ConfidenceScorer）
- 配置：`mma.yaml`

#### 外部方法适配（External Wrappers）

通过 adapter 接入外部 memory 系统。均在 `methods.py` 的 `get_method()` 中**懒加载**注册。

| 方法名 | 入口文件 | 外部源 | 一句话描述 |
|---|---|---|---|
| `a_mem` | `a_mem.py` | `benchmark/a-mem/A-mem/` | Agentic Memory System 的 LLMController |
| `memgpt` | `memgpt.py` | `memgpt/upstream/` | MemGPT 自主记忆管理 |
| `gen_agents` | `gen_agents.py` | 需克隆 MemEngine（见 SETUP.md） | Generative Agents 基线 |
| `evermemos` | `evermemos.py` | `evermemos/upstream/` | 四阶段：memcell 提取 → 索引 → 混合检索（BM25+稠密+agentic+重排）→ 生成 |
| `reflexion` | `reflexion_method.py` | `reflexion/memengine/` | MemEngine RFMemory 反射式记忆（Reflector, Utilization, Store, Recall） |
| `simplemem` | `simplemem.py` | `simplemem/upstream/OmniSimpleMem/` | OmniSimpleMem 统一记忆编排器 |
| `memoryos` | `memoryos.py` | `memoryos/MemoryOS/` | 三级记忆架构：短期 → 中期 → 长期，独立 retriever 和 updater |
| `mirix` | `mirix/official.py` | 内部 socket 服务 | JSON-over-TCP 与 MIRIX runtime 通信 |

**外部方法修改三件套**：
1. `config/methods/<name>.yaml` — 方法参数
2. `benchmark/methods.py` — 懒加载注册（`get_method()` 的 lazy_imports 分支）
3. 对应的 adapter 文件或目录（`benchmark/<name>.py` 或 `benchmark/<name>/`）

#### 检索系统（benchmark/retrieval.py）架构

供 `semantic_rag_*` 使用，三种后端均继承 `_BaseRetriever`：

1. **SparseRetriever**（`legacy_sparse`）：TF-IDF + 关键词重叠评分（35% 词汇 + 65% 语义权重），纯本地
2. **DenseTextRetriever**（`dense_text`）：all-MiniLM-L6-v2 编码轮次文本，余弦相似度检索
3. **DenseMultimodalRetriever**（`dense_multimodal`）：联合编码文本（all-MiniLM-L6-v2）+ 图片（SigLIP2/CLIP），加权余弦相似度

公共 API：`select_round_ids_for_qa(dataset, qa, config)` → 返回待纳入历史的 round ID 列表。支持邻域窗口（neighbor_window）扩展和检索器缓存。

## 你在这里做研究时，优先关注的修改点

1. 如果你想研究”memory method 本身”
   - 先看本文档的「所有 memory method 详解」了解全景分类
   - 再看 benchmark/methods.py — 理解抽象接口（HistoryMethod）、类继承层次和注册机制（get_method）
   - 再看 benchmark/retrieval.py — 如果涉及检索式方法（semantic_rag_*）
   - 再看对应 benchmark/<method>.py 或 benchmark/<method>/ 目录 — 具体方法实现
   - 最后看 config/methods/ 下的 YAML 调整实验参数
   - 如果研究 agentic 方法（M2A / MMA），注意它们自行实现 answer() 绕过 router

2. 如果你想研究“评测指标是否合理”
   - 看 benchmark/evaluator.py
   - 看 benchmark/llm_judge.txt
   - 看 score_locked_llm_judge.py

3. 如果你想研究“数据与上下文构造”
   - 看 benchmark/dataset.py
   - 看 benchmark/common.py
   - 看 config/tasks/ 下任务定义

4. 如果你想研究“模型调用或 API 适配”
   - 看 router/ 下对应实现
   - 看 config/models/ 下配置

## 研究 memory method 的最佳阅读顺序

按你的研究方向，建议按下面顺序深入：
1. **本文档的「所有 memory method 详解」** — 先建立全景认知
2. **run_benchmark.py + benchmark/runner.py** — 理解一次实验从配置到结果的完整调度
3. **benchmark/methods.py** — 理解抽象接口（HistoryMethod）、类继承树、注册机制（get_method）
4. **benchmark/retrieval.py + benchmark/embeddings.py** — 如果研究方向涉及检索式方法
5. **对应 method 的具体实现文件**（`benchmark/<method>.py` 或 `benchmark/<method>/`）
6. **benchmark/dataset.py** — 理解数据格式与历史上下文构造（改 method 时重要）
7. **benchmark/evaluator.py** — 理解评分逻辑（验证 method 效果时看）
8. **config/methods/ 下的 YAML** — 调参入口

建立”从配置 → 数据 → memory 构造 → 推理 → 打分 → 输出”的全局认识后，再聚焦到具体的 method 实现细节。

## 常见工作流

### 1. 安装环境

```bash
conda create -n memeye python=3.10 -y
conda activate memeye
pip install -r requirements.txt
```

### 2. 下载并注册数据

```bash
git lfs install
git clone https://huggingface.co/datasets/MemEyeBench/MemEye data
python register_external_data.py --data-root ./data --overwrite
```

### 3. 运行单次评测

```bash
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/full_context_multimodal.yaml
```

### 4. 评测 open-ended 结果

```bash
python score_locked_llm_judge.py \
  --root runs/<model>/open \
  --judge-model gpt-5.2
```

## 开发时要注意

- 这是一个 benchmark 框架，不是通用业务应用；修改时优先关注评测逻辑、配置加载与结果输出。
- 多数实验依赖配置文件（config/methods、config/models、config/tasks）驱动，而不是硬编码。
- 如果要新增任务或方法，通常需要同时更新配置与 benchmark 对应模块。
- 结果文件主要在 runs/ 中；如果要验证效果，优先查看 metrics.json 与 predictions.jsonl。

## 对 Claude Code 的建议

在这个仓库里，优先理解以下问题：
- 这个改动会影响哪一种评测方式（MCQ / Open-ended）？
- 是否涉及某个 memory method 的实现或配置？
- 是否会影响 benchmark 的输出格式或指标计算？

如果你在这里做修改，应该把“评测结果是否正确、配置是否完整、输出是否可复现”作为首要标准。
