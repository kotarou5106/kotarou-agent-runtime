from __future__ import annotations

from pathlib import Path

from agent_runtime.tools.base import Tool
from agent_runtime.tools.filesystem import ListDirTool, ReadFileTool
from agent_runtime.tools.web_fetch import WebFetchTool
from agent_runtime.tools.web_search import WebSearchTool
from agent_runtime.core.net.http import HttpRequester


def build_readonly_research_tools(
    *,
    fetch_requester: HttpRequester,
    allowed_dir: Path | None = None,
    include_list_dir: bool = False,
    multimodal: bool = True,
    vl_available: bool = False,
) -> list[Tool]:
    tools: list[Tool] = [ReadFileTool(allowed_dir=allowed_dir, multimodal=multimodal, vl_available=vl_available)]
    if include_list_dir:
        tools.append(ListDirTool(allowed_dir=allowed_dir))
    tools.append(WebFetchTool(fetch_requester))
    tools.append(WebSearchTool())
    return tools
