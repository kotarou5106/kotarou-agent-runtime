# LangGraph Workflow Orchestration

This document maps the optional LangGraph backend to the existing Kotarou Agent
Runtime implementation. The native backend remains the default; LangGraph is an
alternative orchestration layer for the LLM/tool loop.

## Backend Boundary

- Native backend: `AgentLoop + PassiveTurnPipeline + DefaultReasoner`.
- LangGraph backend: `AgentLoop + PassiveTurnPipeline + LangGraphReasoner`.
- Selection point: `bootstrap/tools.py::_build_loop_deps`.
- Config field: `agent_runtime/config_models.py::OrchestrationConfig`.
- Config loader: `agent_runtime/config.py::_load_orchestration_config`.

LangGraph does not replace session management, memory retrieval, knowledge
retrieval, prompt rendering, after-reasoning persistence, or outbound dispatch.
Those remain owned by the existing passive runtime:

- `agent_runtime/core/passive_turn.py::PassiveTurnPipeline.run`
- `agent_runtime/lifecycle/phases/before_turn.py`
- `agent_runtime/lifecycle/phases/before_reasoning.py`
- `agent_runtime/lifecycle/phases/prompt_render.py`
- `agent_runtime/lifecycle/phases/after_reasoning.py`
- `agent_runtime/lifecycle/phases/after_turn.py`

LangGraph currently owns the inner LLM/tool loop that used to live in
`DefaultReasoner.run`.

## State

LangGraph state is defined in:

- `agent_runtime/langgraph_runtime/state.py::LangGraphAgentState`
- `agent_runtime/langgraph_runtime/state.py::initial_state`

Important state groups:

- Turn identity: `trace_id`, `session_key`, `channel`, `chat_id`.
- Model context: `messages`, `request_time_iso`.
- Tool visibility: `disabled_tools`, `visible_names`, `visible_order`,
  `preloaded_tools`, `preloaded_tool_order`.
- Loop control: `iteration`, `max_iterations`, `status`,
  `termination_reason`, `error`.
- LLM result: `current_content`, `current_thinking`, `current_tool_calls`,
  `provider_fields`.
- Tool result: `tool_batch`, `current_tool_results`, `tools_used`,
  `tools_unlocked`, `tool_chain`.
- Runtime metrics: `input_samples`, `cache_prompt_tokens`,
  `cache_hit_tokens`, `cache_seen`, `streamed`.
- Final output: `reply`, `thinking`.
- Human-in-the-loop: `pending_interrupt`.

The state intentionally stores checkpoint-friendly values such as lists rather
than Python sets.

## Nodes

The graph is built in:

- `agent_runtime/langgraph_runtime/reasoner.py::LangGraphReasoner._ensure_graph`

Current nodes:

| Node | Function | Responsibility |
| --- | --- | --- |
| `before_step` | `LangGraphReasoner._before_step_node` | Increment iteration, run existing `BeforeStep` phase, record token estimate, detect max iteration / early stop. |
| `llm_reasoning` | `LangGraphReasoner._llm_reasoning_node` | Call `LLMProvider.chat` once with current messages and visible tool schemas. |
| `tool_risk_gate` | `LangGraphReasoner._tool_risk_gate_node` | Apply interrupt policy before high-risk tool execution. |
| `tool_execution` | `LangGraphReasoner._tool_execution_node` | Execute tool batch through `ToolExecutor.execute` and `ToolRegistry.execute`; append tool results to messages. |
| `after_tool_step` | `LangGraphReasoner._after_tool_step_node` | Run existing `AfterStep` phase after a tool batch. |
| `finalize` | `LangGraphReasoner._finalize_node` | Append final assistant message, run final `AfterStep`, set reply/thinking. |
| `summarize_incomplete` | `LangGraphReasoner._summarize_incomplete_node` | Use existing incomplete-progress summarizer for max iteration, early stop, or tool loop guard exits. |

Existing runtime functions wrapped by these nodes:

- `agent_runtime/provider.py::LLMProvider.chat`
- `agent_runtime/tool_hooks/executor.py::ToolExecutor.execute`
- `agent_runtime/tools/registry.py::ToolRegistry.execute`
- `agent_runtime/tool_runtime.py::append_assistant_tool_calls`
- `agent_runtime/tool_runtime.py::append_tool_result`
- `agent_runtime/lifecycle/phases/before_step.py::default_before_step_modules`
- `agent_runtime/lifecycle/phases/after_step.py::default_after_step_modules`

## Edges

Static edges in `LangGraphReasoner._ensure_graph`:

```text
before_step -> llm_reasoning
tool_risk_gate -> tool_execution
finalize -> END
summarize_incomplete -> END
```

These edges are static because the next phase is deterministic once the current
node has succeeded.

## Conditional Edges

Conditional routing functions:

- `LangGraphReasoner._route_after_llm`
- `LangGraphReasoner._route_after_tool_execution`
- `LangGraphReasoner._route_after_step`

Conditional routes:

```text
llm_reasoning
  -> tool_risk_gate        when LLM returned tool_calls
  -> finalize              when LLM returned final content
  -> llm_reasoning         when model produced thinking but no final content
  -> summarize_incomplete  when previous node marked summarize

tool_execution
  -> after_tool_step       after normal tool execution
  -> summarize_incomplete  when tool loop guard or another stop condition fires

after_tool_step
  -> before_step           continue the LLM/tool loop
  -> summarize_incomplete  when AfterStep asks to stop
```

This is the main reason LangGraph is useful here: LLM reasoning and tool
execution are no longer hidden in one monolithic `while` loop.

## Checkpoint

Checkpoint helpers:

- `agent_runtime/langgraph_runtime/checkpoint.py::CheckpointerResource`
- `agent_runtime/langgraph_runtime/checkpoint.py::build_async_checkpointer`
- `agent_runtime/langgraph_runtime/checkpoint.py::build_checkpointer`

Runtime usage:

- `LangGraphReasoner._resolve_checkpointer`
- `LangGraphReasoner._ensure_graph`

Behavior:

- `checkpoint_enabled = true`: prefer `AsyncSqliteSaver` for async graph
  execution and keep its async context manager alive through the reasoner.
- `checkpoint_enabled = false`: use `InMemorySaver`, so checkpoint semantics
  still exist in tests and demos without durable sqlite persistence.
- If sqlite checkpoint setup is unavailable, fall back to `InMemorySaver`.

Important detail: `SqliteSaver.from_conn_string(...)` and
`AsyncSqliteSaver.from_conn_string(...)` return context managers in current
LangGraph versions. The runtime enters the context manager and stores the
resource so the saver is not closed after graph compilation.

## Interrupt Policy

Interrupt policy is defined in:

- `agent_runtime/langgraph_runtime/interrupts.py::ToolInterruptPolicy`
- `agent_runtime/langgraph_runtime/interrupts.py::build_interrupt_payload`
- `agent_runtime/langgraph_runtime/interrupts.py::normalize_resume_decision`

The gate is applied in:

- `agent_runtime/langgraph_runtime/reasoner.py::LangGraphReasoner._tool_risk_gate_node`

Default approval targets:

- Tools with risk `write`.
- Tools with risk `external-side-effect`.
- Explicit tool names: `shell`, `write_file`, `delete_file`, `memorize`,
  `forget_memory`, `schedule`, `message_push`, `spawn`.

The interrupt happens before `ToolExecutor.execute`, so high-risk side effects
are not performed until the graph is resumed with approval.

## Native Compatibility

Native compatibility is preserved by construction:

- Default config is `backend = "native"`.
- `LangGraphReasoner` implements the same `Reasoner` contract as
  `DefaultReasoner`.
- `PassiveTurnPipeline` still owns context preparation and final persistence.
- Tool hooks still run through `ToolExecutor`.
- `ToolRegistry` and existing tool schemas are reused.

## Demo

A self-contained demo is available at:

- `scripts/langgraph_dummy_tool_demo.py`

It sets `OrchestrationConfig(backend="langgraph")`, creates a fake LLM provider,
registers a safe dummy tool, and runs a full:

```text
LLM -> Tool -> LLM -> Final
```

flow through `LangGraphReasoner`.
