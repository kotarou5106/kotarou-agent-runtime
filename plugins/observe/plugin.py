from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from contextlib import suppress
from typing import Protocol, cast, runtime_checkable

from agent_runtime.plugins import Plugin
from agent_runtime.events.events_lifecycle import (
    ToolCallCompleted,
    ToolCallStarted,
    TraceReplayEvent,
    TurnCommitted,
)
from agent_runtime.core.memory.events import MemoryWritten, RetrievalCompleted

from .retention import run_retention_if_needed
from .writer import TraceWriter

logger = logging.getLogger("plugin.observe")


@runtime_checkable
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


class ObservePlugin(Plugin):
    name = "observe"

    async def initialize(self) -> None:
        workspace = self.context.workspace
        if workspace is None:
            logger.warning("observe 插件缺少 workspace，跳过加载")
            return

        self._writer = TraceWriter(workspace / "observe" / "observe.db")
        self._writer_task = asyncio.create_task(
            self._writer.run(),
            name="observe_writer",
        )
        self._retention_task = asyncio.create_task(
            run_retention_if_needed(workspace / "observe" / "observe.db"),
            name="observe_retention",
        )
        self.context.event_bus.on(TurnCommitted, self._observe_turn_committed)
        self.context.event_bus.on(TraceReplayEvent, self._observe_trace_replay)
        self.context.event_bus.on(ToolCallStarted, self._observe_tool_started)
        self.context.event_bus.on(ToolCallCompleted, self._observe_tool_completed)
        self.context.event_bus.on(RetrievalCompleted, self._observe_retrieval)
        self.context.event_bus.on(MemoryWritten, self._observe_memory_written)

    async def terminate(self) -> None:
        for task in (
            getattr(self, "_retention_task", None),
            getattr(self, "_writer_task", None),
        ):
            if task is None:
                continue
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _observe_turn_committed(self, event: TurnCommitted) -> None:
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter):
            return
        _emit_turn_trace(writer, event)

    def _observe_retrieval(self, event: RetrievalCompleted) -> None:
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter):
            return
        writer.emit(_to_rag_query_log(event))

    def _observe_memory_written(self, event: MemoryWritten) -> None:
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter):
            return
        writer.emit(_to_memory_write_trace(event))

    def _observe_trace_replay(self, event: TraceReplayEvent) -> None:
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter):
            return
        writer.emit(_to_trace_replay_log(event))

    def _observe_tool_started(self, event: ToolCallStarted) -> None:
        trace_id = event.trace_id or _trace_id_for_session(event.session_key)
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter) or not trace_id:
            return
        from .events import TraceReplayLog

        writer.emit(
            TraceReplayLog(
                trace_id=trace_id,
                session_key=event.session_key,
                channel=event.channel,
                chat_id=event.chat_id,
                phase="tool",
                event_type="tool.started",
                title=event.tool_name,
                payload={
                    "iteration": event.iteration,
                    "call_id": event.call_id,
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                },
            )
        )

    def _observe_tool_completed(self, event: ToolCallCompleted) -> None:
        trace_id = event.trace_id or _trace_id_for_session(event.session_key)
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter) or not trace_id:
            return
        from .events import TraceReplayLog

        writer.emit(
            TraceReplayLog(
                trace_id=trace_id,
                session_key=event.session_key,
                channel=event.channel,
                chat_id=event.chat_id,
                phase="tool",
                event_type="tool.completed",
                title=event.tool_name,
                payload={
                    "iteration": event.iteration,
                    "call_id": event.call_id,
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                    "final_arguments": event.final_arguments,
                    "status": event.status,
                    "result_preview": event.result_preview,
                },
            )
        )


def _emit_turn_trace(writer: _ObserveWriter, event: TurnCommitted) -> None:
    from .events import TurnTrace as TurnTraceEvent

    post_reply_budget = event.post_reply_budget
    react_stats = event.react_stats
    tool_chain = event.tool_chain_raw
    tool_chain_json = (
        json.dumps(_slim_tool_chain(tool_chain), ensure_ascii=False)
        if tool_chain
        else None
    )
    tool_calls = _slim_tool_calls(tool_chain)
    writer.emit(
        TurnTraceEvent(
            source="agent",
            session_key=event.session_key,
            user_msg=event.persisted_user_message,
            llm_output=event.assistant_response,
            raw_llm_output=event.raw_reply,
            meme_tag=event.meme_tag,
            meme_media_count=event.meme_media_count,
            tool_calls=tool_calls,
            tool_chain_json=tool_chain_json,
            history_window=post_reply_budget.get("history_window"),
            history_messages=post_reply_budget.get("history_messages"),
            history_chars=post_reply_budget.get("history_chars"),
            history_tokens=post_reply_budget.get("history_tokens"),
            prompt_tokens=post_reply_budget.get("prompt_tokens"),
            next_turn_baseline_tokens=post_reply_budget.get(
                "next_turn_baseline_tokens"
            ),
            react_iteration_count=react_stats.get("iteration_count"),
            react_input_sum_tokens=react_stats.get("turn_input_sum_tokens"),
            react_input_peak_tokens=react_stats.get("turn_input_peak_tokens"),
            react_final_input_tokens=react_stats.get("final_call_input_tokens"),
            react_cache_prompt_tokens=react_stats.get("cache_prompt_tokens"),
            react_cache_hit_tokens=react_stats.get("cache_hit_tokens"),
        )
    )
    trace_id = str(event.extra.get("trace_id") or "")
    if trace_id:
        from .events import TraceReplayLog

        writer.emit(
            TraceReplayLog(
                trace_id=trace_id,
                session_key=event.session_key,
                channel=event.channel,
                chat_id=event.chat_id,
                phase="turn",
                event_type="turn.committed",
                title="Turn committed",
                payload={
                    "input_message": event.input_message,
                    "assistant_response": event.assistant_response,
                    "raw_reply": event.raw_reply,
                    "tools_used": event.tools_used,
                    "thinking": event.thinking,
                    "post_reply_budget": event.post_reply_budget,
                    "react_stats": event.react_stats,
                    "tool_chain": event.tool_chain_raw,
                },
            )
        )
    logger.info(
        "[observe] turn_trace 已入队 session=%s tool_calls=%d",
        event.session_key,
        len(tool_calls),
    )


def _to_rag_query_log(event: RetrievalCompleted):
    from .events import RagHitLog, RagQueryLog

    return RagQueryLog(
        caller="passive",
        session_key=event.session_key,
        query=event.query,
        orig_query=event.orig_query,
        aux_queries=list(event.aux_queries),
        hits=[
            RagHitLog(
                item_id=hit.item_id,
                memory_type=hit.memory_type,
                score=hit.score,
                summary=hit.summary[:120],
                injected=hit.injected,
                confidence_label=hit.confidence_label,
                forced=hit.forced,
            )
            for hit in event.hits
        ],
        injected_count=event.injected_count,
        route_decision=event.route_decision,
        error=event.error,
    )


def _to_memory_write_trace(event: MemoryWritten):
    from .events import MemoryWriteTrace

    return MemoryWriteTrace(
        session_key=event.session_key,
        source_ref=event.source_ref,
        action=event.action,
        memory_type=event.memory_type,
        item_id=event.item_id,
        summary=event.summary,
        superseded_ids=list(event.superseded_ids),
        error=event.error,
    )


def _to_trace_replay_log(event: TraceReplayEvent):
    from .events import TraceReplayLog

    return TraceReplayLog(
        trace_id=event.trace_id,
        session_key=event.session_key,
        channel=event.channel,
        chat_id=event.chat_id,
        phase=event.phase,
        event_type=event.event_type,
        title=event.title,
        payload=event.payload,
        source=event.source,
    )


def _trace_id_for_session(session_key: str) -> str:
    # Tool lifecycle events do not carry trace_id yet; using session_key keeps
    # them queryable until the next richer event contract lands.
    return session_key


def _slim_tool_calls(tool_chain: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(call.get("name", "")),
            "args": str(call.get("arguments", ""))[:300],
            "result": str(call.get("result", ""))[:500],
        }
        for group in tool_chain
        for call in _group_calls(group)
    ]


def _slim_tool_chain(tool_chain: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "text": str(group.get("text") or ""),
            "calls": [
                {
                    "name": str(call.get("name", "")),
                    "args": str(call.get("arguments", ""))[:800],
                    "result": str(call.get("result", ""))[:1200],
                }
                for call in _group_calls(group)
            ],
        }
        for group in tool_chain
    ]


def _group_calls(group: dict[str, object]) -> list[dict[str, object]]:
    calls = group.get("calls")
    if not isinstance(calls, list):
        return []
    raw_calls = cast(list[object], calls)
    out: list[dict[str, object]] = []
    for call in raw_calls:
        if isinstance(call, Mapping):
            mapping = cast(Mapping[object, object], call)
            out.append(
                {
                    str(key): value
                    for key, value in mapping.items()
                    if isinstance(key, str)
                }
            )
    return out
