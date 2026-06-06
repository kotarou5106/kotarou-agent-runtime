from __future__ import annotations

import copy
import json
from typing import Any

from agent_runtime.provider import LLMResponse


def _json_chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


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
        captured["messages_json_chars"] = _json_chars(messages)
        captured["tools_schema_json_chars"] = _json_chars(tools)
        self.calls.append(captured)
        if not self._responses:
            raise AssertionError("ScriptedProvider.chat called more than expected")
        return self._responses.pop(0)
