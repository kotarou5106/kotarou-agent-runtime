from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from evaluation_system.harness.compare import BackendComparisonRunner
from evaluation_system.harness.cost import CostProbe, CostReport
from evaluation_system.harness.registry import get_scenario, list_scenarios
from evaluation_system.harness.report import Report
from evaluation_system.harness.runner import HarnessRunner
from evaluation_system.harness.scenario import AssertionSpec, HarnessConfig, Scenario
from agent_runtime.context import ContextBuilder
from agent_runtime.core.memory.engine import MemoryToolProfile, MemoryToolSpec
from agent_runtime.core.types import ContextRequest
from agent_runtime.provider import LLMResponse
from agent_runtime.tools.filesystem import ListDirTool, ReadFileTool
from agent_runtime.tools.meta import register_common_meta_tools, register_memory_meta_tools
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.web_fetch import WebFetchTool
from agent_runtime.tools.web_search import WebSearchTool


def _run(coro):
    return asyncio.run(coro)


def test_harness_registry_lists_builtin_scenarios() -> None:
    names = list_scenarios()

    assert "safe_tool_call" in names
    assert "shell_requires_interrupt" in names
    assert "langgraph_tool_loop" in names
    assert "safe_tool_call_report" in names
    assert "backend_consistency_tool_loop" in names
    assert "token_budget_comparison" in names


def test_safe_tool_call_passes_native_backend() -> None:
    run = _run(
        HarnessRunner().run(
            get_scenario("safe_tool_call"),
            HarnessConfig(backend="native"),
        )
    )

    assert run.error is None
    assert run.passed
    assert run.final_answer == "safe tool done"
    assert run.tools_used == ["dummy_safe_tool"]
    assert [item.passed for item in run.assertion_results] == [True, True, True]
    assert [event.type for event in run.trace_events] == [
        "run_started",
        "llm_call",
        "llm_call",
        "tool_call",
        "tool_result",
        "final_answer",
        "run_finished",
    ]


def test_safe_tool_call_passes_langgraph_backend() -> None:
    run = _run(
        HarnessRunner().run(
            get_scenario("langgraph_tool_loop"),
            HarnessConfig(backend="langgraph"),
        )
    )

    assert run.error is None
    assert run.passed
    assert run.metadata["orchestration"] == "langgraph"
    assert run.tools_used == ["dummy_safe_tool"]
    assert len(run.llm_calls) == 2


def test_shell_requires_interrupt_records_risk_gate_without_execution() -> None:
    run = _run(
        HarnessRunner().run(
            get_scenario("shell_requires_interrupt"),
            HarnessConfig(backend="langgraph", interrupt_high_risk_tools=True),
        )
    )

    assert run.error is None
    assert run.passed
    assert any(
        event.type in {"interrupt.required", "interrupt_required"}
        and event.payload.get("tool_name") == "shell"
        for event in run.trace_events
    )
    assert [item.passed for item in run.assertion_results] == [True, True]


def test_report_generates_json_and_markdown(tmp_path) -> None:
    run = _run(
        HarnessRunner().run(
            get_scenario("safe_tool_call_report"),
            HarnessConfig(backend="native"),
        )
    )
    report = Report(run)

    data = json.loads(report.to_json())
    markdown = report.to_markdown()
    json_path = report.write_json(tmp_path / "safe_tool_call_report.json")
    md_path = report.write_markdown(tmp_path / "safe_tool_call_report.md")

    assert data["scenario_name"] == "safe_tool_call_report"
    assert data["backend"] == "native"
    assert data["passed"] is True
    assert data["tool_call_count"] == 1
    assert data["prompt_chars"] > 0
    assert data["tool_schema_chars"] > 0
    assert data["trace_events"][0]["event_type"] == "run_started"
    assert "Agent Evaluation Report" in markdown
    assert "safe_tool_call_report" in json_path.read_text(encoding="utf-8")
    assert "Trace Events" in md_path.read_text(encoding="utf-8")


def test_backend_comparison_tool_loop_both_passed_and_consistent() -> None:
    comparison = _run(
        BackendComparisonRunner().run(
            get_scenario("backend_consistency_tool_loop"),
            assertions=[
                AssertionSpec("both_passed"),
                AssertionSpec("backend_consistency"),
                AssertionSpec("tool_call_count_delta_lte", 0),
            ],
        )
    )

    assert comparison.passed
    assert comparison.runs["native"].passed
    assert comparison.runs["langgraph"].passed
    assert comparison.runs["native"].tools_used == ["dummy_safe_tool"]
    assert comparison.runs["langgraph"].tools_used == ["dummy_safe_tool"]
    assert [item.passed for item in comparison.assertion_results] == [True, True, True]


def test_token_budget_comparison_accepts_small_delta() -> None:
    comparison = _run(
        BackendComparisonRunner().run(
            get_scenario("token_budget_comparison"),
            assertions=[
                AssertionSpec("both_passed"),
                AssertionSpec("token_delta_lte", 256),
            ],
        )
    )

    assert comparison.passed
    token_result = comparison.assertion_results[1]
    assert token_result.kind == "token_delta_lte"
    assert token_result.evidence["delta"] <= 256


def test_cost_snapshot_and_assertions_are_recorded() -> None:
    scenario = get_scenario("safe_tool_call")
    scenario.assertions.extend(
        [
            AssertionSpec("max_system_chars", 100),
            AssertionSpec("max_messages_json_chars", 1200),
            AssertionSpec("max_tools_schema_chars", 1200),
            AssertionSpec("max_tool_count", 1),
            AssertionSpec("max_estimated_input_tokens", 800),
            AssertionSpec(
                "cost_regression_lte",
                0,
                {"field": "estimated_input_tokens", "baseline": 800},
            ),
        ]
    )

    run = _run(HarnessRunner().run(scenario, HarnessConfig(backend="native")))
    snapshot = CostProbe().snapshot_run(run)

    assert run.passed
    assert snapshot.model == "harness-fake-model"
    assert snapshot.backend == "native"
    assert snapshot.tool_count == 1
    assert snapshot.tools_schema_chars > 0
    assert snapshot.estimated_input_tokens > 0


def test_cost_report_generates_json_and_markdown(tmp_path) -> None:
    run = _run(
        HarnessRunner().run(
            get_scenario("safe_tool_call"),
            HarnessConfig(backend="native"),
        )
    )
    snapshot = CostProbe().snapshot_run(run)
    before = type(snapshot)(
        system_chars=11782,
        messages_chars=12457,
        messages_json_chars=12934,
        tools_schema_chars=16635,
        tool_count=19,
        always_on_tool_count=19,
        estimated_input_tokens=(12934 + 16635) // 3,
        max_tokens=8192,
        model="deepseek-v4-flash",
        backend="native",
    )
    report = CostReport(snapshot, before=before)

    data = json.loads(report.to_json())
    json_path = report.write_json(tmp_path / "cost.json")
    md_path = report.write_markdown(tmp_path / "cost.md")

    assert data["snapshot"]["tool_count"] == 1
    assert data["delta"]["tools_schema_chars"] < 0
    assert "Top Tool Schemas" in report.to_markdown()
    assert "tools_schema_chars" in json_path.read_text(encoding="utf-8")
    assert "Before / After Delta" in md_path.read_text(encoding="utf-8")


def test_production_prompt_and_tool_schema_cost_regression_after_optimization(tmp_path) -> None:
    messages = _render_production_like_messages(tmp_path)
    registry = _production_like_registry()
    always_on = registry.get_always_on_names()
    tools = registry.get_schemas(names=registry.get_registered_order(always_on))
    call = {
        "messages": messages,
        "tools": tools,
        "model": "deepseek-v4-flash",
        "max_tokens": 8192,
    }

    snapshot = CostProbe().snapshot_provider_call(
        call,
        backend="native",
        always_on_tool_count=len(always_on),
    )

    assert snapshot.system_chars < 7000
    assert snapshot.tools_schema_chars < 9000
    assert snapshot.tool_count < 19
    assert snapshot.always_on_tool_count < 19
    assert {"tool_search", "recall_memory", "web_search", "web_fetch", "read_file", "list_dir"} <= always_on
    assert {"spawn", "shell", "write_file", "edit_file", "message_push", "fetch_messages", "search_messages"}.isdisjoint(always_on)


class _CostMemory:
    def read_self(self) -> str:
        return ""

    def get_memory_context(self) -> str:
        return ""

    def read_recent_context(self) -> str:
        return ""


class _CostMemoryEngine:
    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile(
            recall=MemoryToolSpec(
                description="Recall relevant long-term memory.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            memorize=MemoryToolSpec(
                description="Write a stable long-term memory.",
                parameters={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
                risk="write",
            ),
            forget=MemoryToolSpec(
                description="Mark incorrect memories as superseded.",
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["ids"],
                },
                risk="write",
            ),
        )

    async def query(self, request: Any) -> Any:
        raise NotImplementedError

    async def mutate(self, request: Any) -> Any:
        raise NotImplementedError

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None


def _render_production_like_messages(tmp_path) -> list[dict[str, Any]]:
    context = ContextBuilder(tmp_path, cast(Any, _CostMemory()))
    rendered = context.render(
        ContextRequest(
            history=[],
            current_message="请简单回复：收到。",
            channel="cli",
            chat_id="cost",
        )
    )
    return rendered.messages


def _production_like_registry() -> ToolRegistry:
    registry = ToolRegistry()
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }
    register_common_meta_tools(
        registry,
        readonly_tools,
        session_store=object(),
    )
    register_memory_meta_tools(registry, cast(Any, _CostMemoryEngine()))
    return registry
