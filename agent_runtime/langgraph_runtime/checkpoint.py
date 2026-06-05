from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_checkpoint_thread_id(session_key: str, trace_id: str = "") -> str:
    trace = str(trace_id or "").strip()
    return f"{session_key}:{trace}" if trace else session_key


@dataclass
class CheckpointerResource:
    saver: Any
    context_manager: Any | None = None
    async_context_manager: Any | None = None

    def close(self) -> None:
        if self.context_manager is None:
            return
        exit_fn = getattr(self.context_manager, "__exit__", None)
        if callable(exit_fn):
            exit_fn(None, None, None)
        self.context_manager = None

    async def aclose(self) -> None:
        if self.async_context_manager is not None:
            exit_fn = getattr(self.async_context_manager, "__aexit__", None)
            if callable(exit_fn):
                await exit_fn(None, None, None)
            self.async_context_manager = None
        self.close()


def build_checkpointer(
    workspace: Path | None = None,
    *,
    persistent: bool = True,
) -> CheckpointerResource:
    """Return a managed LangGraph checkpointer resource.

    The sqlite checkpointer lives in an optional distribution in many LangGraph
    versions. SqliteSaver.from_conn_string() returns a context manager in
    current LangGraph releases, so this helper enters it and keeps the context
    manager alive through CheckpointerResource.
    """
    if not persistent:
        return CheckpointerResource(_build_in_memory_saver())
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        db_path = (workspace or Path.cwd()) / "langgraph_checkpoints.sqlite"
        cm = SqliteSaver.from_conn_string(str(db_path))
        enter = getattr(cm, "__enter__", None)
        if callable(enter):
            return CheckpointerResource(saver=enter(), context_manager=cm)
        return CheckpointerResource(saver=cm)
    except Exception:
        return CheckpointerResource(_build_in_memory_saver())


async def build_async_checkpointer(
    workspace: Path | None = None,
    *,
    persistent: bool = True,
) -> CheckpointerResource:
    if not persistent:
        return CheckpointerResource(_build_in_memory_saver())
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore

        db_path = (workspace or Path.cwd()) / "langgraph_checkpoints.sqlite"
        cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        enter = getattr(cm, "__aenter__", None)
        if callable(enter):
            return CheckpointerResource(
                saver=await enter(),
                async_context_manager=cm,
            )
        return CheckpointerResource(saver=cm)
    except Exception:
        return CheckpointerResource(_build_in_memory_saver())


def _build_in_memory_saver() -> Any:
    try:
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore

        return InMemorySaver()
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore

        return MemorySaver()
