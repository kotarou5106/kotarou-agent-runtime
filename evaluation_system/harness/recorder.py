from __future__ import annotations

import json
from typing import Any

from evaluation_system.harness.scenario import BackendName, TraceEvent


def summarize_text(value: object, limit: int = 160) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class TraceRecorder:
    def __init__(self, *, scenario_id: str, backend: BackendName) -> None:
        self._scenario_id = scenario_id
        self._backend = backend
        self.events: list[TraceEvent] = []

    def run_started(self, input_messages: list[dict[str, Any]]) -> None:
        self._append(
            "run_started",
            step_name="run",
            input_summary=summarize_text(input_messages),
            metadata={"scenario_id": self._scenario_id},
        )

    def llm_call(self, call: dict[str, Any], *, index: int) -> None:
        token_usage = {
            "prompt_chars": int(call.get("messages_json_chars") or 0),
            "tool_schema_chars": int(call.get("tools_schema_json_chars") or 0),
            "total_request_chars": int(call.get("messages_json_chars") or 0)
            + int(call.get("tools_schema_json_chars") or 0),
        }
        self._append(
            "llm_call",
            node_name="llm_reasoning",
            step_name=f"llm_call:{index}",
            input_summary=summarize_text(call.get("messages") or []),
            output_summary=f"tools={call.get('tool_count', 0)}",
            token_usage=token_usage,
            metadata={
                "model": call.get("model"),
                "message_count": call.get("message_count"),
                "tool_count": call.get("tool_count"),
            },
        )

    def tool_call(self, call: dict[str, Any], *, group_index: int, call_index: int) -> None:
        name = str(call.get("name") or "")
        self._append(
            "tool_call",
            node_name="tool_execution",
            step_name=f"tool:{group_index}:{call_index}",
            tool_name=name,
            input_summary=summarize_text(call.get("arguments") or {}),
            metadata={"call_id": call.get("call_id")},
        )

    def tool_result(
        self,
        call: dict[str, Any],
        *,
        group_index: int,
        call_index: int,
    ) -> None:
        name = str(call.get("name") or "")
        self._append(
            "tool_result",
            node_name="tool_execution",
            step_name=f"tool_result:{group_index}:{call_index}",
            tool_name=name,
            output_summary=summarize_text(call.get("result") or ""),
            metadata={
                "call_id": call.get("call_id"),
                "status": call.get("status"),
                "final_arguments": call.get("final_arguments"),
            },
        )

    def interrupt_required(self, payload: dict[str, Any]) -> None:
        self._append(
            "interrupt_required",
            node_name="tool_risk_gate",
            step_name="interrupt",
            tool_name=str(payload.get("tool_name") or ""),
            input_summary=summarize_text(payload.get("arguments") or {}),
            metadata=dict(payload),
        )

    def checkpoint_saved(self, metadata: dict[str, Any] | None = None) -> None:
        self._append(
            "checkpoint_saved",
            node_name="checkpoint",
            step_name="checkpoint",
            metadata=dict(metadata or {}),
        )

    def final_answer(self, content: str) -> None:
        self._append(
            "final_answer",
            step_name="final",
            output_summary=summarize_text(content),
            metadata={"content": content},
        )

    def run_finished(self, *, passed: bool, error: str | None = None) -> None:
        self._append(
            "run_finished",
            step_name="run",
            output_summary="passed" if passed else "failed",
            metadata={"passed": passed, "error": error},
        )

    def extend(self, events: list[TraceEvent]) -> None:
        self.events.extend(events)

    def _append(
        self,
        event_type: str,
        *,
        node_name: str = "",
        step_name: str = "",
        tool_name: str = "",
        input_summary: str = "",
        output_summary: str = "",
        token_usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(metadata or {})
        self.events.append(
            TraceEvent(
                event_type,
                payload,
                backend=self._backend,
                node_name=node_name,
                step_name=step_name,
                tool_name=tool_name,
                input_summary=input_summary,
                output_summary=output_summary,
                token_usage=dict(token_usage or {}),
                metadata=dict(metadata or {}),
            )
        )


def build_run_trace_events(
    *,
    scenario_id: str,
    backend: BackendName,
    input_messages: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    tool_chain: list[dict[str, Any]],
    preflight_events: list[TraceEvent],
    final_answer: str,
    passed: bool,
    error: str | None,
    checkpoint_enabled: bool,
) -> list[TraceEvent]:
    recorder = TraceRecorder(scenario_id=scenario_id, backend=backend)
    recorder.run_started(input_messages)
    for event in preflight_events:
        if event.type in {"interrupt.required", "interrupt_required"}:
            recorder.interrupt_required(event.payload)
        else:
            recorder.extend([event])
    if checkpoint_enabled and backend == "langgraph":
        recorder.checkpoint_saved({"mode": "in_memory_or_sqlite"})
    for index, call in enumerate(llm_calls, 1):
        recorder.llm_call(call, index=index)
    for group_index, group in enumerate(tool_chain, 1):
        for call_index, call in enumerate(group.get("calls") or [], 1):
            recorder.tool_call(call, group_index=group_index, call_index=call_index)
            recorder.tool_result(call, group_index=group_index, call_index=call_index)
    recorder.final_answer(final_answer)
    recorder.run_finished(passed=passed, error=error)
    return recorder.events
