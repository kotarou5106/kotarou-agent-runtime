from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class LangGraphAgentState(TypedDict):
    trace_id: str
    session_key: str
    channel: str
    chat_id: str
    messages: list[dict[str, Any]]
    request_time_iso: str
    disabled_tools: list[str]
    visible_names: list[str] | None
    visible_order: list[str] | None
    preloaded_tools: list[str] | None
    preloaded_tool_order: list[str]
    iteration: int
    max_iterations: int
    input_samples: list[int]
    tools_used: list[str]
    tools_unlocked: list[str]
    tool_chain: list[dict[str, Any]]
    current_content: str | None
    current_thinking: str | None
    current_tool_calls: list[Any]
    provider_fields: dict[str, Any]
    cache_prompt_tokens: int
    cache_hit_tokens: int
    cache_seen: bool
    streamed: bool
    reply: str
    thinking: str | None
    status: str
    termination_reason: str
    error: str
    tool_batch: tuple[dict[str, Any], ...]
    current_tool_results: list[dict[str, Any]]
    pending_interrupt: NotRequired[dict[str, Any]]


def initial_state(
    *,
    messages: list[dict[str, Any]],
    trace_id: str = "",
    session_key: str = "",
    channel: str = "",
    chat_id: str = "",
    request_time_iso: str = "",
    disabled_tools: set[str] | None = None,
    visible_names: set[str] | None = None,
    visible_order: list[str] | None = None,
    preloaded_tools: set[str] | None = None,
    preloaded_tool_order: list[str] | None = None,
    max_iterations: int = 10,
) -> LangGraphAgentState:
    return {
        "trace_id": trace_id,
        "session_key": session_key,
        "channel": channel,
        "chat_id": chat_id,
        "messages": list(messages),
        "request_time_iso": request_time_iso,
        "disabled_tools": sorted(set(disabled_tools or set())),
        "visible_names": (
            sorted(set(visible_names)) if visible_names is not None else None
        ),
        "visible_order": list(visible_order) if visible_order is not None else None,
        "preloaded_tools": (
            sorted(set(preloaded_tools)) if preloaded_tools is not None else None
        ),
        "preloaded_tool_order": list(preloaded_tool_order or []),
        "iteration": -1,
        "max_iterations": int(max_iterations),
        "input_samples": [],
        "tools_used": [],
        "tools_unlocked": [],
        "tool_chain": [],
        "current_content": None,
        "current_thinking": None,
        "current_tool_calls": [],
        "provider_fields": {},
        "cache_prompt_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_seen": False,
        "streamed": False,
        "reply": "",
        "thinking": None,
        "status": "running",
        "termination_reason": "",
        "error": "",
        "tool_batch": (),
        "current_tool_results": [],
    }
