from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from agent_runtime.provider import LLMResponse
from agent_runtime.tools.base import Tool

BackendName = Literal["native", "langgraph"]


@dataclass(frozen=True)
class HarnessConfig:
    backend: BackendName = "native"
    model: str = "harness-fake-model"
    max_iterations: int = 4
    max_tokens: int = 512
    tool_search_enabled: bool = False
    checkpoint_enabled: bool = False
    interrupt_high_risk_tools: bool = True
    disabled_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AssertionSpec:
    kind: str
    value: Any = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    id: str
    input_messages: list[dict[str, Any]]
    provider_responses: list[LLMResponse]
    tools: list[Tool] = field(default_factory=list)
    tool_risks: dict[str, str] = field(default_factory=dict)
    always_on_tools: frozenset[str] | None = None
    assertions: list[AssertionSpec] = field(default_factory=list)
    description: str = ""
    backends: tuple[BackendName, ...] = ("native", "langgraph")


@dataclass(frozen=True)
class TraceEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    backend: BackendName | str = ""
    node_name: str = ""
    step_name: str = ""
    tool_name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        return self.type


@dataclass
class AssertionResult:
    kind: str
    passed: bool
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    scenario_id: str
    backend: BackendName
    final_answer: str
    trace_events: list[TraceEvent]
    tool_chain: list[dict[str, Any]]
    tools_used: list[str]
    llm_calls: list[dict[str, Any]]
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.error is None and all(item.passed for item in self.assertion_results)


@dataclass
class ComparisonResult:
    scenario_id: str
    runs: dict[BackendName, AgentRun]
    assertion_results: list[AssertionResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.assertion_results)
