# Kotarou Agent Runtime

Kotarou Agent Runtime 是一个 Personal AI Agent Runtime / 个人智能体运行时系统。它关注的不是“把大模型接到一个聊天窗口”，而是把对话、记忆、工具、插件、知识检索、主动任务、可观测性和评测放进同一套可运行的工程链路里。

这个项目适合展示一个 Agent Runtime 如何长期运行：用户消息进入系统后，会经过上下文构建、记忆检索、知识检索、prompt 组装、LLM 推理、工具调用、流式回复、记忆沉淀和事件记录；后台系统还能调度主动任务，并通过 Dashboard 观察运行时状态。

## 项目定位

Kotarou Agent Runtime 的目标是实现一个可扩展的个人 AI Agent 底座，而不是一个普通 RAG Demo 或单轮问答机器人。

它解决的问题包括：

- 如何让 Agent 在多轮对话中保持稳定上下文，而不是每轮重新开始。
- 如何把长期记忆、近期上下文和检索结果分层管理，避免所有信息无差别塞进 prompt。
- 如何让 LLM 在需要时发现并调用工具，并把工具结果带回推理链路。
- 如何通过插件扩展运行时能力，而不是把所有功能硬编码进主循环。
- 如何让后台任务、主动提醒、知识库检索和用户对话共享同一套 runtime 基础设施。
- 如何用 Dashboard、事件日志和评测系统观察 Agent 的行为质量。

## 核心能力

- **Multi-turn Conversation**: 支持会话状态、上下文构建、流式响应、对话历史持久化和运行时事件记录。
- **Long-term Memory**: 将稳定用户事实、Agent 自我模型、时间线事件、近期摘要和待归档事实拆分存储。
- **Tool Calling**: 提供工具注册、工具发现、工具执行、结果回传、错误处理和工具调用边界。
- **Plugin System**: 插件可以注册工具、监听 lifecycle、扩展 Dashboard panel，并参与 runtime 的扩展点。
- **RAG / Knowledge Retrieval**: 知识库内容经过加载、切分、索引、检索和注入，作为 `knowledge_context` 进入 prompt。
- **Proactive Tasks**: 后台任务可以基于信息源、规则、记忆和运行时状态判断是否需要主动触达用户。
- **Dashboard Observability**: 提供 session、prompt section、memory、plugin、proactive task、tool call 等运行时视图。
- **Evaluation System**: 包含 LongMemEval、PersonaMem、RAG 相关评测和 benchmark runtime。
- **Safety / Permission Boundary**: 工具调用、shell 行为、循环保护、undo/recovery 等模块共同约束运行边界。

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

这里的 RAG 不是孤立功能，而是 Agent Runtime 主链路的一部分。知识检索结果会和 memory context、recent context、tool context 一起参与 LLM reasoning；Dashboard 和事件记录也可以帮助观察检索内容是否正确进入了上下文。

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
- `pytest` 1236 passed.
- Dashboard build passed.
- Project is under active refactoring.

本次 README 重写只调整项目说明文档，没有修改 runtime 代码逻辑。

## Interview Notes

面试中可以重点展开：

- **Agent Loop**: 用户消息如何经过 context、memory、knowledge、prompt、LLM、tool 和 response streaming。
- **Memory Consolidation**: `PENDING.md`、`HISTORY.md`、`RECENT_CONTEXT.md` 与 `MEMORY.md` 的分层写入策略。
- **Tool Calling**: 工具注册、发现、执行、结果回传、异常处理和权限边界。
- **RAG Context Injection**: 知识库检索结果如何作为 `knowledge_context` 进入 prompt，而不是和长期记忆混在一起。
- **Proactive Scheduling**: 后台任务如何结合状态、规则、记忆和 delivery 做主动触达。
- **Observability**: Dashboard 如何查看 prompt section、memory、tool call、plugin 和任务状态。
- **Evaluation**: 如何用 LongMemEval、PersonaMem 和 RAG 评测验证 Agent Runtime 的长期能力。
