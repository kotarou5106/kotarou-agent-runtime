from typing import Any, cast
import pytest
from agent_runtime.tools.base import Tool
from agent_runtime.tools.filesystem import ListDirTool, ReadFileTool
from agent_runtime.tools.meta import (
    META_TOOLBOX_NAMES,
    build_meta_toolbox_prompt,
    register_common_meta_tools,
    register_memory_meta_tools,
)
from agent_runtime.tools.message_push import MessagePushTool
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.web_fetch import WebFetchTool
from agent_runtime.tools.web_search import WebSearchTool
from agent_runtime.core.memory.engine import MemoryToolProfile, MemoryToolSpec


class _MemoryEngineStub:
    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile(
            recall=MemoryToolSpec(
                description="test",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            ),
            forget=MemoryToolSpec(
                description="test",
                parameters={"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]},
                risk="write",
            ),
        )

    async def query(self, request):
        raise NotImplementedError

    async def mutate(self, request):
        raise NotImplementedError

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def execute(self, **kwargs):
        return ""


def test_meta_toolbox_prompt_contains_grouped_overview():
    prompt = build_meta_toolbox_prompt()

    assert "MetaToolBox" in prompt
    assert "[Read]" in prompt
    assert "recall_memory" in prompt
    assert "message_push" in prompt
    assert "write_file" in prompt


def test_register_meta_tool_helpers_mark_expected_tools_always_on():
    tools = ToolRegistry()
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    push_tool = register_common_meta_tools(
        tools,
        readonly_tools,
        session_store=object(),
    )
    register_memory_meta_tools(
        tools,
        cast(Any, _MemoryEngineStub()),
    )

    always_on = tools.get_always_on_names()
    assert isinstance(push_tool, MessagePushTool)
    assert {"tool_search", "recall_memory", "web_search", "web_fetch", "read_file", "list_dir"} <= always_on
    assert {"shell", "write_file", "edit_file", "message_push", "fetch_messages", "search_messages", "forget_memory"}.isdisjoint(always_on)
    assert set(META_TOOLBOX_NAMES) - {"memorize"} <= tools.get_registered_names()


def test_register_memory_meta_tools_rejects_duplicate_names():
    tools = ToolRegistry()

    register_memory_meta_tools(tools, cast(Any, _MemoryEngineStub()))

    with pytest.raises(ValueError, match="重复注册"):
        register_memory_meta_tools(tools, cast(Any, _MemoryEngineStub()))
