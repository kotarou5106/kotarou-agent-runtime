from __future__ import annotations

from typing import Any

from agent_runtime.tools.base import Tool


class DummyTool(Tool):
    def __init__(
        self,
        name: str = "dummy_safe_tool",
        *,
        result: str | None = None,
        description: str | None = None,
    ) -> None:
        self._name = name
        self._result = result or f"{name}-ok"
        self._description = description or name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "Optional test value.",
                }
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        value = kwargs.get("value")
        if value:
            return f"{self._result}:{value}"
        return self._result
