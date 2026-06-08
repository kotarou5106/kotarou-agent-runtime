from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, cast

from agent_runtime import core
from agent_runtime.core import passive_support as support
from agent_runtime.core.passive_turn import (
    DefaultReasoner,
    _disabled_tools_from_msg,
    _is_tool_loop_guard_denial,
)
from agent_runtime.core.types import LLMToolCall, ReasonerResult
from agent_runtime.langgraph_runtime.checkpoint import (
    CheckpointerResource,
    build_async_checkpointer,
)
from agent_runtime.langgraph_runtime.interrupts import (
    ToolInterruptPolicy,
    build_interrupt_payload,
    normalize_resume_decision,
)
from agent_runtime.langgraph_runtime.state import LangGraphAgentState, initial_state
from agent_runtime.lifecycle.types import AfterStepCtx, BeforeStepInput
from agent_runtime.provider import ContentSafetyError, ContextLengthError
from agent_runtime.tool_hooks import ToolExecutionRequest
from agent_runtime.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    tool_call_batch_snapshot,
)
from agent_runtime.tools.base import normalize_tool_result
from agent_runtime.tools.tool_search import ToolSearchTool

if False:  # pragma: no cover
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class LangGraphReasoner(DefaultReasoner):
    """LangGraph orchestration backend for the existing passive reasoner.

    This class intentionally implements the same public Reasoner contract as
    DefaultReasoner. The rest of the native PassiveTurnPipeline still owns
    session prepare, memory/RAG retrieval, prompt rendering, persistence, and
    outbound dispatch.
    """

    def __init__(
        self,
        *args: Any,
        workspace=None,
        interrupt_policy: ToolInterruptPolicy | None = None,
        checkpointer: Any | None = None,
        checkpoint_persistent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._workspace = workspace
        self._interrupt_policy = interrupt_policy or ToolInterruptPolicy()
        self._checkpointer = checkpointer
        self._checkpoint_persistent = bool(checkpoint_persistent)
        self._checkpointer_resource: CheckpointerResource | None = (
            checkpointer if isinstance(checkpointer, CheckpointerResource) else None
        )
        self._graph: Any | None = None

    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        disabled_tools: set[str] | None = None,
        trace_id: str = "",
    ) -> ReasonerResult:
        graph = await self._ensure_graph()
        visible_names: set[str] | None = None
        visible_order: list[str] | None = None
        disabled = set(disabled_tools or set())
        preloaded_order = list(preloaded_tool_order or [])
        if self._tool_search_enabled:
            always_on = self._tools.get_always_on_names()
            visible_names = (always_on | (preloaded_tools or set())) - disabled
            visible_order = self._tools.get_registered_order(always_on - disabled)
            seen_visible = set(visible_order)
            for name in preloaded_order or sorted(preloaded_tools or set()):
                if name in visible_names and name not in seen_visible:
                    visible_order.append(name)
                    seen_visible.add(name)

        state = initial_state(
            messages=initial_messages,
            trace_id=trace_id,
            session_key=tool_event_session_key,
            channel=tool_event_channel,
            chat_id=tool_event_chat_id,
            request_time_iso=request_time.isoformat() if request_time else "",
            disabled_tools=disabled,
            visible_names=visible_names,
            visible_order=visible_order,
            preloaded_tools=preloaded_tools,
            preloaded_tool_order=preloaded_order,
            max_iterations=self._llm_config.max_iterations,
        )
        config = {
            "configurable": {
                "thread_id": tool_event_session_key or trace_id or "langgraph-turn"
            }
        }
        try:
            final_state = await graph.ainvoke(state, config=config)
        except ContentSafetyError:
            raise
        except ContextLengthError:
            raise
        except TimeoutError:
            raise
        except Exception:
            logger.exception("LangGraphReasoner.run failed")
            raise
        return self._state_to_result(cast(LangGraphAgentState, final_state))

    async def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        try:
            from langgraph.graph import END, StateGraph  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "LangGraph mode requires installing the optional dependency: "
                "pip install langgraph langgraph-checkpoint-sqlite"
            ) from exc

        builder = StateGraph(LangGraphAgentState)
        builder.add_node("before_step", self._before_step_node)
        builder.add_node("llm_reasoning", self._llm_reasoning_node)
        builder.add_node("tool_risk_gate", self._tool_risk_gate_node)
        builder.add_node("tool_execution", self._tool_execution_node)
        builder.add_node("after_tool_step", self._after_tool_step_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("summarize_incomplete", self._summarize_incomplete_node)

        builder.set_entry_point("before_step")
        builder.add_edge("before_step", "llm_reasoning")
        builder.add_conditional_edges(
            "llm_reasoning",
            self._route_after_llm,
            {
                "tools": "tool_risk_gate",
                "final": "finalize",
                "retry_empty": "llm_reasoning",
                "summarize": "summarize_incomplete",
            },
        )
        builder.add_edge("tool_risk_gate", "tool_execution")
        builder.add_conditional_edges(
            "tool_execution",
            self._route_after_tool_execution,
            {
                "after_step": "after_tool_step",
                "summarize": "summarize_incomplete",
            },
        )
        builder.add_conditional_edges(
            "after_tool_step",
            self._route_after_step,
            {
                "continue": "before_step",
                "summarize": "summarize_incomplete",
            },
        )
        builder.add_edge("finalize", END)
        builder.add_edge("summarize_incomplete", END)
        checkpointer = await self._resolve_checkpointer()
        self._graph = builder.compile(
            checkpointer=checkpointer,
        )
        return self._graph

    async def _resolve_checkpointer(self) -> Any:
        if isinstance(self._checkpointer, CheckpointerResource):
            self._checkpointer_resource = self._checkpointer
            return self._checkpointer.saver
        if self._checkpointer is not None:
            return self._checkpointer
        self._checkpointer_resource = await build_async_checkpointer(
            self._workspace,
            persistent=self._checkpoint_persistent,
        )
        return self._checkpointer_resource.saver

    async def _before_step_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        iteration = int(state["iteration"]) + 1
        if state["max_iterations"] > 0 and iteration >= state["max_iterations"]:
            return {
                "iteration": iteration,
                "status": "summarize",
                "termination_reason": "max_iterations",
            }
        step_ctx = await self._before_step.run(
            BeforeStepInput(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                messages=state["messages"],
                visible_names=(
                    set(state["visible_names"])
                    if state["visible_names"] is not None
                    else None
                ),
            )
        )
        input_samples = list(state["input_samples"])
        input_samples.append(step_ctx.input_tokens_estimate)
        if step_ctx.early_stop:
            return {
                "iteration": iteration,
                "input_samples": input_samples,
                "status": "summarize",
                "termination_reason": "early_stop",
                "reply": step_ctx.early_stop_reply or "",
            }
        return {
            "iteration": iteration,
            "input_samples": input_samples,
            "status": "running",
        }

    async def _llm_reasoning_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        schema_names: list[str] | set[str] | None = (
            list(state["visible_order"]) if state["visible_order"] is not None else None
        )
        disabled = set(state["disabled_tools"])
        if schema_names is None and disabled:
            schema_names = self._tools.get_registered_names() - disabled
        elif schema_names is not None:
            schema_names = [name for name in schema_names if name not in disabled]

        started = time.perf_counter()
        response = await self._llm.provider.chat(
            messages=state["messages"],
            tools=self._tools.get_schemas(names=schema_names),
            model=self._llm_config.model,
            max_tokens=self._llm_config.max_tokens,
            tool_choice="auto",
        )
        _ = int((time.perf_counter() - started) * 1000)
        cache_prompt = state["cache_prompt_tokens"]
        cache_hit = state["cache_hit_tokens"]
        cache_seen = state["cache_seen"]
        if response.cache_prompt_tokens is not None:
            cache_seen = True
            cache_prompt += response.cache_prompt_tokens
            cache_hit += response.cache_hit_tokens or 0
        messages = list(state["messages"])
        current_tool_calls = list(response.tool_calls)
        if current_tool_calls:
            append_assistant_tool_calls(
                messages,
                content=response.content,
                tool_calls=current_tool_calls,
                provider_fields=response.provider_fields,
            )
        return {
            "messages": messages,
            "current_content": response.content,
            "current_thinking": response.thinking,
            "current_tool_calls": current_tool_calls,
            "provider_fields": dict(response.provider_fields),
            "cache_prompt_tokens": cache_prompt,
            "cache_hit_tokens": cache_hit,
            "cache_seen": cache_seen,
            "streamed": bool(state["streamed"] or response.content),
            "status": "needs_tool" if current_tool_calls else "final",
        }

    def _route_after_llm(self, state: LangGraphAgentState) -> str:
        if state["status"] == "summarize":
            return "summarize"
        if state["current_tool_calls"]:
            return "tools"
        if not state["current_content"] and state["current_thinking"]:
            messages = list(state["messages"])
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "user",
                    "content": "你刚才只输出了思考过程，没有给出正式回复。请直接回复用户，不要重复思考。",
                }
            )
            state["messages"] = messages
            return "retry_empty"
        return "final"

    async def _tool_risk_gate_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        try:
            from langgraph.types import interrupt  # type: ignore
        except Exception:
            interrupt = None
        approvals: dict[str, Any] = {}
        if interrupt is None:
            return {"pending_interrupt": {}}
        for tool_call in state["current_tool_calls"]:
            if not self._interrupt_policy.requires_approval(self._tools, tool_call):
                continue
            payload = build_interrupt_payload(
                tool_call=tool_call,
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                trace_id=state["trace_id"],
            )
            approvals[tool_call.id] = interrupt(payload)
        return {"pending_interrupt": approvals}

    async def _tool_execution_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        messages = list(state["messages"])
        tool_batch = tool_call_batch_snapshot(state["current_tool_calls"])
        iter_calls: list[dict[str, Any]] = []
        tools_used = list(state["tools_used"])
        tools_unlocked = list(state["tools_unlocked"])
        visible_names = (
            set(state["visible_names"]) if state["visible_names"] is not None else None
        )
        visible_order = (
            list(state["visible_order"]) if state["visible_order"] is not None else None
        )
        approvals = state.get("pending_interrupt", {})

        for tool_batch_index, tool_call in enumerate(state["current_tool_calls"]):
            call_args = dict(tool_call.arguments)
            if isinstance(approvals, dict) and tool_call.id in approvals:
                action, edited_args = normalize_resume_decision(approvals[tool_call.id])
                if action == "reject":
                    result = "工具调用被人工审批拒绝。请基于已有信息继续，或向用户说明需要该操作才能完成。"
                    append_tool_result(
                        messages,
                        tool_call_id=tool_call.id,
                        content=result,
                        tool_name=tool_call.name,
                    )
                    iter_calls.append(
                        {
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "status": "denied",
                            "arguments": dict(tool_call.arguments),
                            "final_arguments": dict(tool_call.arguments),
                            "result": result,
                            "human_interrupted": True,
                        }
                    )
                    continue
                if action == "edit" and edited_args is not None:
                    call_args = dict(edited_args)

            if tool_call.name in set(state["disabled_tools"]):
                result = (
                    f"工具 '{tool_call.name}' 在当前后台任务中不可用。"
                    "请直接返回要发送的最终内容，不要主动推送。"
                )
                append_tool_result(
                    messages,
                    tool_call_id=tool_call.id,
                    content=result,
                    tool_name=tool_call.name,
                )
                iter_calls.append(
                    {
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "status": "blocked",
                        "arguments": dict(tool_call.arguments),
                        "final_arguments": dict(tool_call.arguments),
                        "result": result,
                    }
                )
                continue

            if visible_names is not None and tool_call.name not in visible_names:
                result = (
                    f"工具 '{tool_call.name}' 当前未加载（schema 不可见）。"
                    f"请先调用 tool_search(query=\"select:{tool_call.name}\") 加载，"
                    "然后再调用该工具。不要放弃当前任务。"
                )
                append_tool_result(messages, tool_call_id=tool_call.id, content=result)
                iter_calls.append(
                    {
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "status": "blocked",
                        "arguments": dict(tool_call.arguments),
                        "final_arguments": dict(tool_call.arguments),
                        "result": result,
                    }
                )
                continue

            if (
                tool_call.name == "tool_search"
                and visible_names is not None
                and isinstance(self._tool_search_tool, ToolSearchTool)
            ):
                self._tool_search_tool.set_excluded_names(
                    visible_names | set(state["disabled_tools"])
                )
            tool_started = time.perf_counter()
            exec_result = await self._tool_executor.execute(
                ToolExecutionRequest(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=call_args,
                    source="passive",
                    session_key=state["session_key"],
                    channel=state["channel"],
                    chat_id=state["chat_id"],
                    tool_batch=tool_batch,
                    tool_batch_index=tool_batch_index,
                ),
                self._tools.execute,
            )
            tool_latency_ms = int((time.perf_counter() - tool_started) * 1000)
            if exec_result.status == "success":
                tools_used.append(tool_call.name)
            normalized = normalize_tool_result(exec_result.output)
            append_tool_result(
                messages,
                tool_call_id=tool_call.id,
                content=exec_result.output,
                tool_name=tool_call.name,
            )
            self._record_tool_policy_reward(
                tool_name=tool_call.name,
                status=exec_result.status,
                latency_ms=tool_latency_ms,
                exec_result=exec_result,
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
            )
            if (
                exec_result.status == "success"
                and tool_call.name == "tool_search"
                and visible_names is not None
            ):
                newly_unlocked = [
                    name
                    for name in self._discovery.unlock_names_from_result(normalized.text)
                    if name not in visible_names
                    and name not in set(state["disabled_tools"])
                ]
                if newly_unlocked:
                    visible_names.update(newly_unlocked)
                    tools_unlocked.extend(newly_unlocked)
                    if visible_order is not None:
                        seen_visible = set(visible_order)
                        for name in newly_unlocked:
                            if name not in seen_visible:
                                visible_order.append(name)
                                seen_visible.add(name)
            iter_calls.append(
                {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "status": exec_result.status,
                    "arguments": dict(tool_call.arguments),
                    "final_arguments": dict(exec_result.final_arguments),
                    "pre_hook_trace": [
                        {
                            "hook_name": item.hook_name,
                            "event": item.event,
                            "matched": item.matched,
                            "decision": item.decision,
                            "reason": item.reason,
                            "extra_message": item.extra_message,
                        }
                        for item in exec_result.pre_hook_trace
                    ],
                    "post_hook_trace": [
                        {
                            "hook_name": item.hook_name,
                            "event": item.event,
                            "matched": item.matched,
                            "decision": item.decision,
                            "reason": item.reason,
                            "extra_message": item.extra_message,
                        }
                        for item in exec_result.post_hook_trace
                    ],
                    "result": normalized.preview(),
                }
            )
            if _is_tool_loop_guard_denial(exec_result):
                return {
                    "messages": messages,
                    "tools_used": tools_used,
                    "tools_unlocked": tools_unlocked,
                    "visible_names": (
                        sorted(visible_names) if visible_names is not None else None
                    ),
                    "visible_order": visible_order,
                    "current_tool_results": iter_calls,
                    "status": "summarize",
                    "termination_reason": "tool_call_loop",
                }

        tool_chain = list(state["tool_chain"])
        group: dict[str, Any] = {"text": state["current_content"], "calls": iter_calls}
        if state["current_thinking"] is not None:
            group["reasoning_content"] = state["current_thinking"]
        tool_chain.append(group)
        return {
            "messages": messages,
            "tools_used": tools_used,
            "tools_unlocked": tools_unlocked,
            "visible_names": (
                sorted(visible_names) if visible_names is not None else None
            ),
            "visible_order": visible_order,
            "tool_chain": tool_chain,
            "current_tool_results": iter_calls,
            "status": "running",
        }

    def _route_after_tool_execution(self, state: LangGraphAgentState) -> str:
        return "summarize" if state["status"] == "summarize" else "after_step"

    async def _after_tool_step_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        pressure_tokens = support.estimate_messages_tokens(state["messages"])
        after_step = await self._after_step.run(
            AfterStepCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=state["iteration"],
                context_tokens_estimate=pressure_tokens,
                tools_called=tuple(tc.name for tc in state["current_tool_calls"]),
                partial_reply=state["current_content"] or "",
                tools_used_so_far=tuple(state["tools_used"]),
                tool_chain_partial=tuple(state["tool_chain"]),
                partial_thinking=state["current_thinking"],
                has_more=True,
            )
        )
        if after_step.early_stop:
            return {
                "status": "summarize",
                "termination_reason": after_step.early_stop_reason or "after_step",
            }
        return {"status": "running"}

    def _route_after_step(self, state: LangGraphAgentState) -> str:
        return "summarize" if state["status"] == "summarize" else "continue"

    async def _finalize_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        messages = list(state["messages"])
        messages.append({"role": "assistant", "content": state["current_content"]})
        _ = await self._after_step.run(
            AfterStepCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=state["iteration"],
                context_tokens_estimate=support.estimate_messages_tokens(messages),
                tools_called=(),
                partial_reply=state["current_content"] or "",
                tools_used_so_far=tuple(state["tools_used"]),
                tool_chain_partial=tuple(state["tool_chain"]),
                partial_thinking=state["current_thinking"],
                has_more=False,
            )
        )
        return {
            "messages": messages,
            "reply": state["current_content"] or "（无响应）",
            "thinking": state["current_thinking"],
            "status": "final",
        }

    async def _summarize_incomplete_node(
        self,
        state: LangGraphAgentState,
    ) -> dict[str, Any]:
        if state.get("reply") and state["termination_reason"] == "early_stop":
            return {"status": "final"}
        summary = await self._summarize_incomplete_progress(
            state["messages"],
            reason=state["termination_reason"] or "langgraph_stop",
            iteration=state["iteration"],
            tools_used=state["tools_used"],
        )
        return {"reply": summary, "thinking": None, "status": "final"}

    def _state_to_result(self, state: LangGraphAgentState) -> ReasonerResult:
        invocations: list[LLMToolCall] = []
        for group in state["tool_chain"]:
            for call in group.get("calls") or []:
                args = call.get("arguments")
                invocations.append(
                    LLMToolCall(
                        id=str(call.get("call_id", "") or ""),
                        name=str(call.get("name", "") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        react_stats = {
            "iteration_count": len(state["input_samples"]),
            "turn_input_sum_tokens": sum(state["input_samples"]),
            "turn_input_peak_tokens": max(state["input_samples"], default=0),
            "final_call_input_tokens": (
                state["input_samples"][-1] if state["input_samples"] else 0
            ),
        }
        if state["cache_seen"]:
            react_stats["cache_prompt_tokens"] = state["cache_prompt_tokens"]
            react_stats["cache_hit_tokens"] = state["cache_hit_tokens"]
        return ReasonerResult(
            reply=state["reply"] or "（无响应）",
            invocations=invocations,
            thinking=state["thinking"],
            streamed=state["streamed"],
            metadata={
                "tools_used": list(state["tools_used"]),
                "tools_unlocked": list(state["tools_unlocked"]),
                "tool_chain": list(state["tool_chain"]),
                "visible_names": (
                    set(state["visible_names"])
                    if state["visible_names"] is not None
                    else None
                ),
                "react_stats": react_stats,
                "orchestration": "langgraph",
                "termination_reason": state["termination_reason"],
            },
        )
