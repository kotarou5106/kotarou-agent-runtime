# Personal AI Agent OS

Personal AI Agent Runtime 是一个面向个人使用的 AI Agent 系统，支持多轮对话、长期记忆、工具调用、插件扩展、主动推送、后台任务和可观测 Dashboard。

它的目标不是做一个单一聊天机器人，而是做一套可以长期运行、持续理解你、主动处理信息、并允许你逐步扩展能力的 Personal AI Agent OS。

## 目录结构

项目按 Agent 能力模式组织：

- `app/`：启动入口与应用装配
- `agent_runtime/`：Agent 运行时、prompt、lifecycle、LLM/tool loop
- `conversation_patterns/`：被动对话、主动对话、消息流模式
- `memory_system/`：长期记忆、语义记忆、RAG、consolidation
- `tool_system/`：工具注册、发现、执行、MCP、hook
- `planning_system/`：调度、后台任务、drift、资源优化
- `proactive_system/`：主动推送、信号源、优先级、presence、delivery
- `multi_agent_system/`：subagent、peer agent、A2A 协作
- `connectors/`：Telegram、QQ、CLI、IPC、MCP、HTTP 等外部连接
- `safety_system/`：guardrails、shell safety、loop guard、undo/recovery
- `evaluation_system/`：LongMemEval、PersonaMem、评测运行时
- `dashboard/`：Dashboard 后端、前端、静态构建产物
- `storage/`：session、memory、proactive 等本地状态
- `docs/`：架构说明和使用手册
- `scripts/`：维护脚本、Docker 脚本、调试脚本
- `tests/`：测试用例

## Quickstart

需要 Python 3.12。

```bash
git clone <this-repo>
cd personal-agent-os
uv venv
uv pip install -r requirements.txt
```

没有 uv 可以先安装：

```bash
pip install uv
```

初始化：

```bash
uv run python main.py setup
uv run python main.py init
```

示例配置：

```toml
[llm]
provider = "deepseek"

[llm.main]
model = "deepseek-v4-flash"
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"
enable_thinking = true
multimodal = false

[llm.fast]
model = "qwen-flash"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[llm.vl]
model = "qwen-vl-plus"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[memory]
enabled = true
engine = ""

[memory.embedding]
model = "text-embedding-v3"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[channels.telegram]
token = "123456:ABC..."
allow_from = ["your_username"]
```

启动：

```bash
uv run python main.py
```

也可以使用应用入口：

```bash
uv run python app/main.py
```

## 系统全景

```text
用户消息
  -> channel / connector
  -> event bus
  -> agent runtime
  -> memory retrieval
  -> prompt assembly
  -> LLM + tool loop
  -> outbound reply

proactive tick
  -> source polling
  -> ranking / judge
  -> dedupe / gate
  -> message generation
  -> delivery / ACK

idle time
  -> drift skill selection
  -> background tool loop
  -> silent finish or one proactive message
```

## 核心能力

**多轮对话**：支持被动消息入口、上下文组装、流式回复、工具调用和对话历史持久化。

**长期记忆**：通过 Markdown 记忆层和向量记忆层记录事实、偏好、近期上下文，并在回复前检索注入。

**工具系统**：内置文件、记忆、搜索、消息推送、调度、MCP 等工具，并支持插件注册新工具。

**插件扩展**：插件可以介入 lifecycle phase、监听事件、拦截工具调用、注册工具、扩展 Dashboard panel。

**主动推送**：系统定期拉取 alert/content/context 信息源，结合记忆和规则判断是否需要主动联系用户。

**后台任务**：Drift 在没有主动内容可推时执行用户定义的 `SKILL.md` 后台任务。

**Dashboard**：提供会话、记忆、proactive 状态、工具调用、插件面板等可观测入口。

## 文档

| 主题 | 文档 |
| --- | --- |
| 当前目录结构 | [docs/architecture/personal_agent_os_structure.md](./docs/architecture/personal_agent_os_structure.md) |
| 主动推送 | [docs/handbook/proactive-guide.md](./docs/handbook/proactive-guide.md) |
| Drift 后台任务 | [docs/handbook/drift-guide.md](./docs/handbook/drift-guide.md) |
| 记忆系统 | [docs/handbook/memory-markdown.md](./docs/handbook/memory-markdown.md) |
| 插件系统 | [docs/handbook/plugins-tutorial.md](./docs/handbook/plugins-tutorial.md) |

## 常用命令

```bash
uv run python main.py --help
uv run python main.py cli
uv run python main.py dashboard
pytest tests/
```

## 工作区

默认运行时数据在 `~/.kotarou/workspace/`。
