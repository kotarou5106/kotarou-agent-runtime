from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evaluation_system.harness.scenario import AgentRun, BackendName


def json_chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def content_chars(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    return json_chars(value)


@dataclass(frozen=True)
class ToolSchemaCost:
    name: str
    chars: int


@dataclass(frozen=True)
class PromptCostSnapshot:
    system_chars: int
    messages_chars: int
    messages_json_chars: int
    tools_schema_chars: int
    tool_count: int
    always_on_tool_count: int
    estimated_input_tokens: int
    max_tokens: int
    model: str
    backend: BackendName | str
    tool_schema_rankings: list[ToolSchemaCost] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_schema_rankings"] = [
            asdict(item) for item in self.tool_schema_rankings
        ]
        return data


class CostRecorder:
    """Records provider.chat inputs without calling a real provider."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, call: dict[str, Any]) -> PromptCostSnapshot:
        captured = copy.deepcopy(call)
        self.calls.append(captured)
        return snapshot_from_provider_call(captured)


class CostProbe:
    def snapshot_provider_call(
        self,
        call: dict[str, Any],
        *,
        backend: BackendName | str = "",
        always_on_tool_count: int | None = None,
    ) -> PromptCostSnapshot:
        return snapshot_from_provider_call(
            call,
            backend=backend,
            always_on_tool_count=always_on_tool_count,
        )

    def snapshot_run(self, run: AgentRun) -> PromptCostSnapshot:
        if not run.llm_calls:
            return PromptCostSnapshot(
                system_chars=0,
                messages_chars=0,
                messages_json_chars=0,
                tools_schema_chars=0,
                tool_count=0,
                always_on_tool_count=0,
                estimated_input_tokens=0,
                max_tokens=0,
                model="",
                backend=run.backend,
            )
        return self.snapshot_provider_call(
            run.llm_calls[0],
            backend=run.backend,
            always_on_tool_count=len(run.llm_calls[0].get("tools") or []),
        )


def snapshot_from_provider_call(
    call: dict[str, Any],
    *,
    backend: BackendName | str = "",
    always_on_tool_count: int | None = None,
) -> PromptCostSnapshot:
    messages = list(call.get("messages") or [])
    tools = list(call.get("tools") or [])
    system = ""
    if messages and messages[0].get("role") == "system":
        system = str(messages[0].get("content") or "")
    messages_chars = sum(content_chars(item.get("content", "")) for item in messages)
    messages_json_chars = json_chars(messages)
    tools_schema_chars = json_chars(tools)
    rankings = sorted(
        [
            ToolSchemaCost(
                name=str((schema.get("function") or {}).get("name") or ""),
                chars=json_chars(schema),
            )
            for schema in tools
            if isinstance(schema, dict)
        ],
        key=lambda item: item.chars,
        reverse=True,
    )
    estimated = max(1, (messages_json_chars + tools_schema_chars) // 3)
    return PromptCostSnapshot(
        system_chars=len(system),
        messages_chars=messages_chars,
        messages_json_chars=messages_json_chars,
        tools_schema_chars=tools_schema_chars,
        tool_count=len(tools),
        always_on_tool_count=(
            len(tools) if always_on_tool_count is None else always_on_tool_count
        ),
        estimated_input_tokens=estimated,
        max_tokens=int(call.get("max_tokens") or 0),
        model=str(call.get("model") or ""),
        backend=backend or str(call.get("backend") or ""),
        tool_schema_rankings=rankings,
    )


class CostReport:
    def __init__(
        self,
        snapshot: PromptCostSnapshot,
        *,
        before: PromptCostSnapshot | None = None,
        title: str = "Token Cost Report",
    ) -> None:
        self.snapshot = snapshot
        self.before = before
        self.title = title

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "title": self.title,
            "snapshot": self.snapshot.to_dict(),
        }
        if self.before is not None:
            data["before"] = self.before.to_dict()
            data["delta"] = _snapshot_delta(self.before, self.snapshot)
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self, *, top_n: int = 8) -> str:
        snap = self.snapshot
        lines = [
            f"# {self.title}",
            "",
            f"- Backend: `{snap.backend}`",
            f"- Model: `{snap.model}`",
            f"- System chars: {snap.system_chars}",
            f"- Messages chars: {snap.messages_chars}",
            f"- Messages JSON chars: {snap.messages_json_chars}",
            f"- Tools schema chars: {snap.tools_schema_chars}",
            f"- Tool count: {snap.tool_count}",
            f"- Always-on tool count: {snap.always_on_tool_count}",
            f"- Estimated input tokens: {snap.estimated_input_tokens}",
            f"- Max tokens: {snap.max_tokens}",
            "",
            "## Top Tool Schemas",
        ]
        for item in snap.tool_schema_rankings[:top_n]:
            lines.append(f"- `{item.name}`: {item.chars} chars")
        if self.before is not None:
            lines.extend(["", "## Before / After Delta"])
            for key, value in _snapshot_delta(self.before, self.snapshot).items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines).rstrip() + "\n"

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def _snapshot_delta(
    before: PromptCostSnapshot,
    after: PromptCostSnapshot,
) -> dict[str, int]:
    return {
        "system_chars": after.system_chars - before.system_chars,
        "messages_json_chars": after.messages_json_chars - before.messages_json_chars,
        "tools_schema_chars": after.tools_schema_chars - before.tools_schema_chars,
        "tool_count": after.tool_count - before.tool_count,
        "estimated_input_tokens": (
            after.estimated_input_tokens - before.estimated_input_tokens
        ),
    }
