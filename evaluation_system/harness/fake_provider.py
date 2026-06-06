from __future__ import annotations

import copy
from typing import Any

from agent_runtime.provider import LLMResponse
from evaluation_system.harness.cost import snapshot_from_provider_call


class ScriptedProvider:
    """Deterministic provider for offline agent behavior evals."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        captured = copy.deepcopy(kwargs)
        messages = captured.get("messages") or []
        tools = captured.get("tools") or []
        captured["message_count"] = len(messages)
        captured["tool_count"] = len(tools)
        snapshot = snapshot_from_provider_call(captured)
        captured["system_chars"] = snapshot.system_chars
        captured["messages_chars"] = snapshot.messages_chars
        captured["messages_json_chars"] = snapshot.messages_json_chars
        captured["tools_schema_json_chars"] = snapshot.tools_schema_chars
        captured["estimated_input_tokens"] = snapshot.estimated_input_tokens
        self.calls.append(captured)
        if not self._responses:
            raise AssertionError("ScriptedProvider.chat called more than expected")
        return self._responses.pop(0)
