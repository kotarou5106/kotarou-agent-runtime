from __future__ import annotations

import asyncio
import json

from evaluation_system.harness.compare import BackendComparisonRunner
from evaluation_system.harness.registry import get_scenario, list_scenarios
from evaluation_system.harness.report import Report
from evaluation_system.harness.runner import HarnessRunner
from evaluation_system.harness.scenario import AssertionSpec, HarnessConfig


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
