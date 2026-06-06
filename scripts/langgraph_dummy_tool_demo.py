from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_runtime.config_models import OrchestrationConfig
from agent_runtime.core.runtime_support import LLMServices, ToolDiscoveryState
from agent_runtime.langgraph_runtime import LangGraphReasoner
from agent_runtime.langgraph_runtime.interrupts import ToolInterruptPolicy
from agent_runtime.looping.ports import LLMConfig
from agent_runtime.provider import LLMResponse, ToolCall
from agent_runtime.tools.base import Tool
from agent_runtime.tools.registry import ToolRegistry


class DummySafeTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "dummy_safe_tool"

    @property
    def description(self) -> str:
        return "A safe read-only demo tool that echoes a short value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "Value to echo in the demo.",
                }
            },
            "required": ["value"],
        }

    async def execute(self, value: str, **_: Any) -> str:
        self.calls.append({"value": value})
        return f"dummy-result:{value}"


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_dummy_1",
                        name="dummy_safe_tool",
                        arguments={"value": "langgraph"},
                    )
                ],
            ),
            LLMResponse(
                content="Final answer after tool result: dummy-result:langgraph",
                tool_calls=[],
            ),
        ]

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("FakeLLMProvider was called more times than expected")
        return self._responses.pop(0)


async def main() -> None:
    orchestration = OrchestrationConfig(
        backend="langgraph",
        checkpoint_enabled=False,
        interrupt_high_risk_tools=True,
    )
    if orchestration.backend != "langgraph":
        raise RuntimeError("demo must run with backend='langgraph'")

    provider = FakeLLMProvider()
    tools = ToolRegistry()
    dummy_tool = DummySafeTool()
    tools.register(dummy_tool, always_on=True, risk="read-only")

    reasoner = LangGraphReasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(
            model="fake-demo-model",
            max_iterations=4,
            max_tokens=512,
            tool_search_enabled=False,
        ),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        interrupt_policy=ToolInterruptPolicy(enabled=orchestration.interrupt_high_risk_tools),
        checkpoint_persistent=orchestration.checkpoint_enabled,
    )

    result = await reasoner.run(
        [{"role": "user", "content": "Use the dummy safe tool, then answer."}],
        tool_event_session_key="demo:langgraph",
        tool_event_channel="demo",
        tool_event_chat_id="langgraph",
        trace_id="demo-trace",
    )

    print(
        json.dumps(
            {
                "backend": orchestration.backend,
                "checkpoint_enabled": orchestration.checkpoint_enabled,
                "reply": result.reply,
                "tools_used": result.metadata.get("tools_used"),
                "tool_chain": result.metadata.get("tool_chain"),
                "llm_call_count": len(provider.calls),
                "dummy_tool_calls": dummy_tool.calls,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
