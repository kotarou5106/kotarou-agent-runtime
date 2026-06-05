from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

pytest.importorskip("langgraph")

from agent_runtime.core.runtime_support import LLMServices, ToolDiscoveryState
from agent_runtime.langgraph_runtime import LangGraphReasoner
from agent_runtime.langgraph_runtime.interrupts import ToolInterruptPolicy
from agent_runtime.looping.ports import LLMConfig
from agent_runtime.provider import LLMResponse, ToolCall
from agent_runtime.tools.base import Tool
from agent_runtime.tools.registry import ToolRegistry


class _DummyTool(Tool):
    def __init__(self, name: str = "dummy") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"{self._name}-ok"


class _Provider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


def _reasoner(provider: _Provider, tools: ToolRegistry) -> LangGraphReasoner:
    return LangGraphReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        interrupt_policy=ToolInterruptPolicy(enabled=False),
        checkpointer=None,
    )


def test_langgraph_reasoner_runs_real_tool_loop() -> None:
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tool = _DummyTool()
    tools.register(tool, always_on=True)

    result = asyncio.run(
        _reasoner(provider, tools).run([{"role": "user", "content": "hi"}])
    )

    assert result.reply == "final"
    assert result.metadata["orchestration"] == "langgraph"
    assert result.metadata["tools_used"] == ["dummy"]
    assert result.metadata["tool_chain"][0]["calls"][0]["name"] == "dummy"
    assert tool.calls == [{}]


def test_langgraph_reasoner_routes_without_tools_to_finalize() -> None:
    provider = _Provider([LLMResponse(content="plain", tool_calls=[])])
    tools = ToolRegistry()

    result = asyncio.run(
        _reasoner(provider, tools).run([{"role": "user", "content": "hi"}])
    )

    assert result.reply == "plain"
    assert result.metadata["tools_used"] == []
    assert result.metadata["tool_chain"] == []
    assert len(provider.calls) == 1


def test_interrupt_policy_marks_write_and_external_tools() -> None:
    registry = ToolRegistry()
    registry.register(_DummyTool("safe"), always_on=True, risk="read-only")
    registry.register(_DummyTool("danger"), always_on=True, risk="external-side-effect")
    policy = ToolInterruptPolicy(enabled=True)

    assert not policy.requires_approval(registry, ToolCall("c1", "safe", {}))
    assert policy.requires_approval(registry, ToolCall("c2", "danger", {}))
    assert policy.requires_approval(registry, ToolCall("c3", "shell", {}))
