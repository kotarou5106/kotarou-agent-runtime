from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_runtime.tools.registry import ToolRegistry

_DEFAULT_INTERRUPT_TOOLS = frozenset(
    {
        "shell",
        "write_file",
        "delete_file",
        "memorize",
        "forget_memory",
        "schedule",
        "message_push",
        "spawn",
    }
)


class ToolCallLike(Protocol):
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolInterruptPolicy:
    enabled: bool = True
    risk_levels: frozenset[str] = field(
        default_factory=lambda: frozenset({"write", "external-side-effect"})
    )
    tool_names: frozenset[str] = field(default_factory=lambda: _DEFAULT_INTERRUPT_TOOLS)

    def requires_approval(
        self,
        registry: ToolRegistry,
        tool_call: ToolCallLike,
    ) -> bool:
        if not self.enabled:
            return False
        if tool_call.name in self.tool_names:
            return True
        meta = getattr(registry, "_metadata", {}).get(tool_call.name)
        risk = str(getattr(meta, "risk", "") or "")
        return risk in self.risk_levels


def build_interrupt_payload(
    *,
    tool_call: ToolCallLike,
    session_key: str,
    channel: str,
    chat_id: str,
    trace_id: str,
) -> dict[str, Any]:
    return {
        "kind": "tool_approval",
        "tool_name": tool_call.name,
        "arguments": dict(tool_call.arguments),
        "call_id": tool_call.id,
        "session_key": session_key,
        "channel": channel,
        "chat_id": chat_id,
        "trace_id": trace_id,
        "actions": ["approve", "reject", "edit"],
    }


def normalize_resume_decision(value: object) -> tuple[str, dict[str, Any] | None]:
    if isinstance(value, dict):
        action = str(value.get("action") or value.get("decision") or "approve").lower()
        edited = value.get("arguments")
        return action, edited if isinstance(edited, dict) else None
    if isinstance(value, str):
        return value.strip().lower() or "approve", None
    if value is False:
        return "reject", None
    return "approve", None
