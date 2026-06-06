from __future__ import annotations

from collections.abc import Callable

from evaluation_system.harness.scenario import Scenario
from evaluation_system.harness.scenarios import (
    backend_consistency_tool_loop,
    langgraph_tool_loop,
    safe_tool_call,
    safe_tool_call_report,
    shell_requires_interrupt,
    token_budget_comparison,
)

ScenarioFactory = Callable[[], Scenario]

_REGISTRY: dict[str, ScenarioFactory] = {
    "safe_tool_call": safe_tool_call,
    "shell_requires_interrupt": shell_requires_interrupt,
    "langgraph_tool_loop": langgraph_tool_loop,
    "safe_tool_call_report": safe_tool_call_report,
    "backend_consistency_tool_loop": backend_consistency_tool_loop,
    "token_budget_comparison": token_budget_comparison,
}


def list_scenarios() -> list[str]:
    return sorted(_REGISTRY)


def get_scenario(name: str) -> Scenario:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown harness scenario: {name}") from exc


def register_scenario(name: str, factory: ScenarioFactory) -> None:
    if not name.strip():
        raise ValueError("scenario name must not be empty")
    _REGISTRY[name] = factory
