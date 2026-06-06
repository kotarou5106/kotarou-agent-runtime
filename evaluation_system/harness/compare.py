from __future__ import annotations

from evaluation_system.harness.runner import HarnessRunner
from evaluation_system.harness.scenario import (
    AgentRun,
    AssertionResult,
    AssertionSpec,
    BackendName,
    ComparisonResult,
    HarnessConfig,
    Scenario,
)


class BackendComparisonRunner:
    def __init__(self, runner: HarnessRunner | None = None) -> None:
        self._runner = runner or HarnessRunner()

    async def run(
        self,
        scenario: Scenario,
        *,
        base_config: HarnessConfig | None = None,
        assertions: list[AssertionSpec] | None = None,
    ) -> ComparisonResult:
        config = base_config or HarnessConfig()
        runs: dict[BackendName, AgentRun] = {}
        for backend in ("native", "langgraph"):
            runs[backend] = await self._runner.run(
                scenario,
                HarnessConfig(
                    backend=backend,
                    model=config.model,
                    max_iterations=config.max_iterations,
                    max_tokens=config.max_tokens,
                    tool_search_enabled=config.tool_search_enabled,
                    checkpoint_enabled=config.checkpoint_enabled,
                    interrupt_high_risk_tools=config.interrupt_high_risk_tools,
                    disabled_tools=config.disabled_tools,
                ),
            )
        specs = assertions or [
            AssertionSpec("both_passed"),
            AssertionSpec("backend_consistency"),
        ]
        return ComparisonResult(
            scenario_id=scenario.id,
            runs=runs,
            assertion_results=evaluate_comparison_assertions(runs, specs),
            metadata={
                "native_passed": runs["native"].passed,
                "langgraph_passed": runs["langgraph"].passed,
            },
        )


def evaluate_comparison_assertions(
    runs: dict[BackendName, AgentRun],
    assertions: list[AssertionSpec],
) -> list[AssertionResult]:
    return [_evaluate_one(runs, spec) for spec in assertions]


def _evaluate_one(
    runs: dict[BackendName, AgentRun],
    spec: AssertionSpec,
) -> AssertionResult:
    match spec.kind:
        case "both_passed":
            return _both_passed(runs)
        case "backend_consistency":
            return _backend_consistency(runs)
        case "tool_call_count_delta_lte":
            return _tool_call_count_delta_lte(runs, int(spec.value))
        case "token_delta_lte":
            return _token_delta_lte(runs, int(spec.value))
    return AssertionResult(
        kind=spec.kind,
        passed=False,
        message=f"Unknown comparison assertion kind: {spec.kind}",
    )


def _both_passed(runs: dict[BackendName, AgentRun]) -> AssertionResult:
    native = runs["native"].passed
    langgraph = runs["langgraph"].passed
    return AssertionResult(
        kind="both_passed",
        passed=native and langgraph,
        message=f"native={native}, langgraph={langgraph}",
        evidence={"native": native, "langgraph": langgraph},
    )


def _backend_consistency(runs: dict[BackendName, AgentRun]) -> AssertionResult:
    native = runs["native"]
    langgraph = runs["langgraph"]
    native_tools = _tool_names(native)
    langgraph_tools = _tool_names(langgraph)
    native_interrupts = _interrupt_tools(native)
    langgraph_interrupts = _interrupt_tools(langgraph)
    passed = (
        native.passed == langgraph.passed
        and native_tools == langgraph_tools
        and native_interrupts == langgraph_interrupts
    )
    return AssertionResult(
        kind="backend_consistency",
        passed=passed,
        message=(
            f"passed native/langgraph={native.passed}/{langgraph.passed}, "
            f"tools={native_tools}/{langgraph_tools}, "
            f"interrupts={native_interrupts}/{langgraph_interrupts}"
        ),
        evidence={
            "native_tools": native_tools,
            "langgraph_tools": langgraph_tools,
            "native_interrupts": native_interrupts,
            "langgraph_interrupts": langgraph_interrupts,
        },
    )


def _tool_call_count_delta_lte(
    runs: dict[BackendName, AgentRun],
    limit: int,
) -> AssertionResult:
    native = len(_tool_names(runs["native"]))
    langgraph = len(_tool_names(runs["langgraph"]))
    delta = abs(native - langgraph)
    return AssertionResult(
        kind="tool_call_count_delta_lte",
        passed=delta <= limit,
        message=f"tool_call_delta={delta}, limit={limit}",
        evidence={"native": native, "langgraph": langgraph, "delta": delta},
    )


def _token_delta_lte(
    runs: dict[BackendName, AgentRun],
    limit: int,
) -> AssertionResult:
    native = _total_request_chars(runs["native"])
    langgraph = _total_request_chars(runs["langgraph"])
    delta = abs(native - langgraph)
    return AssertionResult(
        kind="token_delta_lte",
        passed=delta <= limit,
        message=f"token_delta={delta}, limit={limit}",
        evidence={"native": native, "langgraph": langgraph, "delta": delta},
    )


def _tool_names(run: AgentRun) -> list[str]:
    return [
        str(call.get("name") or "")
        for group in run.tool_chain
        for call in group.get("calls") or []
        if call.get("name")
    ]


def _interrupt_tools(run: AgentRun) -> list[str]:
    return [
        str(event.tool_name or event.payload.get("tool_name") or "")
        for event in run.trace_events
        if event.type in {"interrupt_required", "interrupt.required"}
    ]


def _total_request_chars(run: AgentRun) -> int:
    total = 0
    for call in run.llm_calls:
        total += int(call.get("messages_json_chars") or 0)
        total += int(call.get("tools_schema_json_chars") or 0)
    return total
