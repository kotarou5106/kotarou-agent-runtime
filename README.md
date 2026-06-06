# Personal AI Agent Runtime

A personal AI Agent runtime with multi-turn chat, long-term memory, tool calling, plugin system, proactive tasks, LangGraph workflow orchestration, evaluation harness, token cost evaluation, and dashboard observability.

Kotarou Agent Runtime 关注的不是“把大模型接到一个聊天窗口”，而是把对话、记忆、工具、插件、知识检索、主动任务、可观测性和评测放进同一套可运行的工程链路里。

## Core Capabilities

- Multi-turn Chat
- Long-term Memory
- Tool Calling
- Plugin System
- Proactive Tasks
- LangGraph Workflow Orchestration
- Hybrid RAG Retrieval
  - Vector Search
  - BM25 Sparse Retrieval
  - Query Rewrite
  - Reciprocal Rank Fusion, RRF
  - Optional LLM Reranking
  - Parent-Child Indexing
  - Citation Validation
- Knowledge RAG Observability
  - Query Rewrite Trace
  - Vector Search Raw Hits
  - BM25 Raw Hits
  - RRF Ranking Trace
  - Reranking Trace
  - Parent-child Expansion Trace
  - Citation Validation Trace
  - Prompt Injection Trace
- Knowledge Retrieval Evaluation
  - BM25 vs Vector vs Hybrid RRF comparison
  - Recall@K
  - Precision@K
  - HitRate@K
  - MRR@K
  - NDCG@K
  - JSON / Markdown evaluation report
- Agent Evaluation & Constraint Harness
- Token Cost Evaluation
- Dashboard Observability

## Engineering Highlights

- **Native / LangGraph dual backend**: native backend remains the default; LangGraph is an optional workflow orchestration backend.
- **LangGraph StateGraph orchestration**: introduced LangGraph workflow orchestration as an optional backend to handle the LLM / tool loop, while reusing the existing context, memory, retrieval, tool registry, and lifecycle modules.
- **Agent Evaluation & Constraint Harness**: supports scenarios, assertions, trace recorder, JSON / Markdown reports, and native-vs-LangGraph backend comparison.
- **Offline token cost evaluation**: captures messages and tool schemas before `provider.chat(...)` without requiring a real API key.
- **Prompt / tool schema optimization**: reduced always-on tools from 19 to 6 while keeping deferred tools discoverable through tool search.
- **Hybrid RAG Retrieval**: rewrites knowledge queries, retrieves with vector search and BM25 sparse retrieval, merges ranked chunks with Reciprocal Rank Fusion, optionally reranks candidates with an LLM reranker, expands child hits to parent context chunks, and injects citation-aware context into the main Agent prompt.
- **Knowledge RAG Observability**: stores lightweight retrieval traces with query variants, vector / BM25 raw hits, RRF rankings, citation validation, prompt injection selections, warnings, errors, and stage timings for dashboard inspection.
- **Knowledge Retrieval Evaluation**: runs lightweight internal retrieval benchmarks comparing BM25, vector search, and Hybrid RRF with Recall@K, Precision@K, HitRate@K, MRR@K, and NDCG@K reports.
- **Offline reproducible tests**: fake provider and dummy tools validate tool-use, interrupt, trace, report, comparison, and cost behavior without real API requests.
- **Test coverage**: full local test suite currently passes with `1284 passed`.

## Test Results

- Full tests: `1284 passed`
- LangGraph runtime tests: passed
- Harness tests: passed
- No real API request required for harness / cost tests

## Token Cost Optimization

| Item | Before | After |
| --- | ---: | ---: |
| system prompt | 11,782 chars | 5,173 chars |
| messages JSON | 12,934 chars | 5,590 chars |
| tools schema JSON | 16,635 chars | 3,316 chars |
| always-on tools | 19 | 6 |
| estimated input tokens | 9,856 | 2,968 |

## Architecture Notes

- LangGraph workflow: `docs/architecture/langgraph_workflow.md`
- Evaluation harness: `docs/architecture/evaluation_harness.md`
- Token cost optimization: `docs/architecture/token_cost_optimization.md`

## Runtime Architecture

```text
User Message
  -> Connector / Channel
  -> Context Builder
  -> Memory Retrieval
  -> Knowledge Retrieval
  -> Prompt Assembly
  -> LLM Reasoning
  -> Tool Calling
  -> Response Streaming
  -> Memory Consolidation
  -> Event Logging
  -> Dashboard
```

这条主链路体现了项目的核心设计：LLM 不是单独运行，而是被放在一个可观测、可扩展、可治理的 runtime 中。记忆和知识检索在推理前进入上下文，工具调用在推理中闭环，回复后的事件与记忆写入又为后续轮次提供状态。

## Main Modules

- `agent_runtime/`: Agent 主运行时，包含 core loop、turn 处理、prompting、retrieval、tool hooks、plugins、lifecycle、background 和 policy。
- `memory_system/`: 长期记忆相关模块，包含 consolidation、learning、markdown memory、memory tools 和 retrieval。
- `knowledge_system/`: RAG 知识系统，包含 loading、chunking、embedding、indexing、retrieval 和 injection。
- `tool_system/`: 工具系统，包含 built-in tools、registry、tool discovery、execution、MCP tools 和 tool hooks。
- `proactive_system/`: 主动任务运行时，负责 delivery 等主动触达链路。
- `connectors/`: 外部连接层，包含 CLI、Telegram、QQ、HTTP、IPC、MCP、providers 和 persistence。
- `dashboard/`: 可观测 Dashboard，包含 backend、frontend、plugin panels 和 static 构建产物。
- `evaluation_system/`: 评测系统，包含 benchmark runtime、LongMemEval、PersonaMem、RAG 评测和 metrics。
- `storage/`: 本地运行时状态，包含 sessions、memory、proactive 和 JSON store。
- `config.example.toml`: 配置模板，运行前复制为 `config.toml` 并填写模型、渠道和记忆相关配置。

项目中还保留了 `app/`、`bootstrap/`、`conversation_patterns/`、`multi_agent_system/`、`planning_system/`、`safety_system/`、`plugins/`、`scripts/`、`tests/` 等目录，用于应用装配、启动初始化、多 Agent 扩展、调度、安全治理、默认插件、调试脚本和测试覆盖。

## Memory Design

Kotarou Agent Runtime 使用 Markdown 记忆层承载长期状态，默认位于 `~/.kotarou/workspace/memory/`。

- `MEMORY.md`: 稳定长期记忆，例如用户偏好、长期事实、重要背景和需要跨会话保留的信息。
- `SELF.md`: Agent 自我模型，用于描述 Agent 的角色边界、表达方式和对用户关系的长期理解。
- `HISTORY.md`: 时间线事件日志，按追加方式保存，可用于检索、审计和记忆沉淀。
- `PENDING.md`: 待归档事实缓冲区，保存从对话中抽取但尚未合并进长期记忆的信息。
- `RECENT_CONTEXT.md`: 近期上下文摘要，用于保留最近关注点、未完成话题和短期状态。

这个设计的重点是分离长期记忆和当前上下文。系统不会把所有历史对话直接塞进 prompt，而是通过近期摘要、长期记忆、事件日志检索和待归档缓冲共同维护上下文压力。这样既能保留长期连续性，也能控制 prompt 成本和噪声。

## RAG / Knowledge System

`knowledge_system/` 负责把外部知识变成可注入的 runtime context。知识内容会经过加载、切分、embedding、索引和检索；当用户问题需要知识库支持时，检索结果会以 `knowledge_context` 的形式进入 prompt assembly。

当前 Knowledge RAG pipeline:

```text
query rewrite
  -> vector retrieval + BM25 retrieval
  -> Reciprocal Rank Fusion, RRF
  -> optional LLM reranking
  -> parent-child expansion
  -> citation-aware prompt injection
  -> main Agent LLM response
  -> retrieval trace for dashboard observability
```

这里的 RAG 不是孤立功能，也不是独立 QA generation chain，而是 Agent Runtime 主链路的一部分。知识检索结果会和 memory context、recent context、tool context 一起参与 LLM reasoning；Dashboard 和事件记录也可以帮助观察检索内容是否正确进入了上下文。

Reranking 是 optional：没有配置 LLM provider 或关闭 reranking 时会使用 no-op reranker，保持原始排序。Parent-child indexing 默认兼容旧数据：child chunks 用于 vector / BM25 检索，parent chunks 用于最终上下文注入；如果旧 chunk 没有 `parent_id` 或找不到 parent，会 fallback 为原 chunk。Trace 中会记录 reranking 输入/输出、parent-child child-to-parent mapping、fallback count 和最终 parent hits。

## Tool Calling

工具系统由 `tool_system/` 和 `agent_runtime/tools`、`agent_runtime/tool_hooks` 等 runtime 模块共同组成。

典型流程是：

```text
Tool Registry
  -> Tool Discovery
  -> LLM Tool Selection
  -> Tool Execution
  -> Result Serialization
  -> Reasoning Continuation
  -> Error / Boundary Handling
```

工具可以来自内置实现、插件注册或 MCP。运行时负责把工具 schema 暴露给 LLM，在模型选择工具后执行调用，把结构化结果回传给推理链路，并在异常、权限或循环风险出现时进入错误处理和安全边界。

## Plugin System

`plugins/` 和 `agent_runtime/plugins/` 提供扩展入口。插件可以注册工具、监听事件、参与 lifecycle phase、扩展 Dashboard panel，或为特定能力提供 runtime hook。

这种设计让能力扩展不必全部堆进 Agent 主循环。主链路保持稳定，插件通过明确接口接入，便于面试中解释模块边界、扩展策略和运行时治理。

## Proactive Runtime

`proactive_system/` 负责主动任务相关能力。它可以围绕外部信息源、用户偏好、当前状态、记忆和规则做后台判断，在满足条件时生成主动触达内容，并通过 connector delivery 发送给用户。

主动能力在这里不是简单定时消息，而是运行时的一条后台链路：它需要读取上下文、判断优先级、避免重复打扰、记录任务状态，并让 Dashboard 能观察相关事件。

## Dashboard and Observability

Dashboard 用于观察 Agent Runtime 的实际行为，而不是只展示静态配置。

可以关注的内容包括：

- session 和消息流状态
- prompt section 与上下文注入结果
- memory 文件和记忆写入状态
- knowledge context 与检索结果
- tool call、tool result 和错误信息
- plugin panel 与插件注册状态
- proactive task、delivery 和后台事件
- evaluation 或调试运行产生的 trace

这部分是项目区别于普通 Demo 的关键：系统不仅能回答问题，也能解释运行时发生了什么。

## Evaluation

`evaluation_system/` 用于验证 Agent 在长期记忆、人格一致性、知识检索和运行时链路上的表现。当前目录中包含：

- `longmemeval/`: 长期记忆评测运行逻辑。
- `personamem/`: 人设和用户画像相关评测。
- `rag/`: 知识检索相关评测。
- `benchmark_runtime/`: 复用生产 runtime 的 benchmark 装配。
- `metrics/`: 评测指标与结果处理。

评测系统的价值在于把“感觉 Agent 更聪明了”变成可重复运行的检查项，尤其适合展示 memory consolidation、RAG context injection 和多轮一致性。

Knowledge Retrieval Evaluation 提供项目内部的轻量检索评测，不做 final answer generation evaluation。它使用固定 dataset 对 BM25、Vector Search 和 Hybrid RRF 进行对比，输出 Recall@K、Precision@K、HitRate@K、MRR@K 和 NDCG@K，并生成 JSON / Markdown 报告。

```bash
python scripts/evaluate_knowledge_retrieval.py
```

默认报告输出到：

- `reports/knowledge_retrieval_eval.json`
- `reports/knowledge_retrieval_eval.md`

## How to Run

需要 Python 3.12 和 `uv`。

```bash
uv venv
uv pip install -r requirements.txt
```

如果还没有 `uv`：

```bash
pip install uv
```

创建配置并初始化 workspace：

```bash
cp config.example.toml config.toml
uv run python main.py setup
uv run python main.py init
```

启动 runtime：

```bash
uv run python main.py
```

启动 Dashboard：

```bash
uv run python main.py dashboard
```

常用检查：

```bash
python -m compileall .
pytest
npm run build
```

Dashboard 前端源码在 `dashboard/frontend/` 下，构建脚本由仓库根目录的 `package.json` 统一管理。

## Current Status

- `compileall` passed.
- `pytest` 1284 passed.
- LangGraph runtime tests passed.
- Evaluation harness and token cost tests passed without real API requests.
- Dashboard build passed.
- Project is under active refactoring.

Kotarou Agent Runtime is still under active development. The current focus is improving memory quality, tool execution reliability, proactive scheduling, and dashboard observability.
