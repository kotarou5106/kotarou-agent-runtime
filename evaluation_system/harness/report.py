from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation_system.harness.recorder import summarize_text
from evaluation_system.harness.scenario import AgentRun, TraceEvent
from evaluation_system.harness.cost import CostProbe


def run_summary(run: AgentRun) -> dict[str, Any]:
    prompt_chars = sum(int(call.get("messages_json_chars") or 0) for call in run.llm_calls)
    tool_schema_chars = sum(
        int(call.get("tools_schema_json_chars") or 0) for call in run.llm_calls
    )
    total_request_chars = prompt_chars + tool_schema_chars
    tool_call_count = sum(len(group.get("calls") or []) for group in run.tool_chain)
    interrupt_count = sum(
        1
        for event in run.trace_events
        if event.type in {"interrupt_required", "interrupt.required"}
    )
    return {
        "scenario_name": run.scenario_id,
        "backend": run.backend,
        "passed": run.passed,
        "failed_assertions": [
            asdict(item) for item in run.assertion_results if not item.passed
        ],
        "passed_assertions": [
            asdict(item) for item in run.assertion_results if item.passed
        ],
        "tool_call_count": tool_call_count,
        "interrupt_count": interrupt_count,
        "prompt_chars": prompt_chars,
        "tool_schema_chars": tool_schema_chars,
        "total_tokens": total_request_chars,
        "final_answer_summary": summarize_text(run.final_answer),
        "error": run.error,
        "cost_snapshot": CostProbe().snapshot_run(run).to_dict(),
    }


class Report:
    def __init__(self, run: AgentRun) -> None:
        self.run = run

    def to_dict(self) -> dict[str, Any]:
        return {
            **run_summary(self.run),
            "trace_events": [_event_to_dict(event) for event in self.run.trace_events],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines = [
            f"# Agent Evaluation Report: {data['scenario_name']}",
            "",
            f"- Backend: `{data['backend']}`",
            f"- Status: {'PASS' if data['passed'] else 'FAIL'}",
            f"- Tool calls: {data['tool_call_count']}",
            f"- Interrupts: {data['interrupt_count']}",
            f"- Prompt chars: {data['prompt_chars']}",
            f"- Tool schema chars: {data['tool_schema_chars']}",
            f"- Total tokens: {data['total_tokens']}",
            f"- Final: {data['final_answer_summary']}",
            "",
            "## Assertions",
        ]
        snapshot = data["cost_snapshot"]
        lines.extend(
            [
                "",
                "## Cost Snapshot",
                f"- System chars: {snapshot['system_chars']}",
                f"- Messages JSON chars: {snapshot['messages_json_chars']}",
                f"- Tools schema chars: {snapshot['tools_schema_chars']}",
                f"- Tool count: {snapshot['tool_count']}",
                f"- Estimated input tokens: {snapshot['estimated_input_tokens']}",
            ]
        )
        for item in self.run.assertion_results:
            marker = "PASS" if item.passed else "FAIL"
            lines.append(f"- {marker} `{item.kind}`: {item.message}")
        lines.extend(["", "## Trace Events"])
        for event in self.run.trace_events:
            label = event.step_name or event.node_name or event.tool_name
            suffix = f" ({label})" if label else ""
            lines.append(f"- `{event.type}`{suffix}: {event.output_summary or event.input_summary}")
        return "\n".join(lines).rstrip() + "\n"

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def _event_to_dict(event: TraceEvent) -> dict[str, Any]:
    return {
        "event_type": event.type,
        "timestamp": event.timestamp,
        "backend": event.backend,
        "node_name": event.node_name,
        "step_name": event.step_name,
        "tool_name": event.tool_name,
        "input_summary": event.input_summary,
        "output_summary": event.output_summary,
        "token_usage": dict(event.token_usage),
        "metadata": dict(event.metadata),
        "payload": dict(event.payload),
    }
