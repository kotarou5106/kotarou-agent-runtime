from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "LangGraphReasoner":
        from agent_runtime.langgraph_runtime.reasoner import LangGraphReasoner

        return LangGraphReasoner
    raise AttributeError(name)

__all__ = ["LangGraphReasoner"]
