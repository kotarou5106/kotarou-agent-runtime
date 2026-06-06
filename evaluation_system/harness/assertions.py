from __future__ import annotations

from evaluation_system.harness.scenario import (
    AgentRun,
    AssertionResult,
    AssertionSpec,
)


def evaluate_assertions(
    run: AgentRun,
    assertions: list[AssertionSpec],
) -> list[AssertionResult]:
    return [_evaluate_one(run, spec) for spec in assertions]


def _evaluate_one(run: AgentRun, spec: AssertionSpec) -> AssertionResult:
    match spec.kind:
        case "required_tool":
            return _required_tool(run, str(spec.value))
        case "forbidden_tool":
            return _forbidden_tool(run, str(spec.value))
        case "max_tool_calls":
            return _max_tool_calls(run, int(spec.value))
        case "max_tokens":
            return _max_tokens(run, int(spec.value))
        case "requires_interrupt":
            return _requires_interrupt(run, str(spec.value))
        case "no_direct_shell_execution":
            return _no_direct_shell_execution(run)
        case "final_contains":
            return _final_contains(run, str(spec.value))
        case "final_not_contains":
            return _final_not_contains(run, str(spec.value))
        case (
            "backend_consistency"
            | "both_passed"
            | "tool_call_count_delta_lte"
            | "token_delta_lte"
        ):
            return AssertionResult(
                kind=spec.kind,
                passed=False,
                message=(
                    f"{spec.kind} is a comparison assertion; "
                    "use BackendComparisonRunner"
                ),
            )
    return AssertionResult(
        kind=spec.kind,
        passed=False,
        message=f"Unknown assertion kind: {spec.kind}",
    )


def _tool_call_names(run: AgentRun) -> list[str]:
    names: list[str] = []
    for group in run.tool_chain:
        for call in group.get("calls") or []:
            names.append(str(call.get("name") or ""))
    return [name for name in names if name]


def _required_tool(run: AgentRun, name: str) -> AssertionResult:
    names = _tool_call_names(run)
    passed = name in names
    return AssertionResult(
        kind="required_tool",
        passed=passed,
        message=f"required tool {name!r} {'was' if passed else 'was not'} called",
        evidence={"tool_calls": names},
    )


def _forbidden_tool(run: AgentRun, name: str) -> AssertionResult:
    names = _tool_call_names(run)
    passed = name not in names
    return AssertionResult(
        kind="forbidden_tool",
        passed=passed,
        message=f"forbidden tool {name!r} {'was not' if passed else 'was'} called",
        evidence={"tool_calls": names},
    )


def _max_tool_calls(run: AgentRun, limit: int) -> AssertionResult:
    names = _tool_call_names(run)
    passed = len(names) <= limit
    return AssertionResult(
        kind="max_tool_calls",
        passed=passed,
        message=f"tool calls={len(names)}, limit={limit}",
        evidence={"tool_calls": names},
    )


def _max_tokens(run: AgentRun, limit: int) -> AssertionResult:
    peak = 0
    for call in run.llm_calls:
        total = int(call.get("messages_json_chars") or 0) + int(
            call.get("tools_schema_json_chars") or 0
        )
        peak = max(peak, total)
    passed = peak <= limit
    return AssertionResult(
        kind="max_tokens",
        passed=passed,
        message=f"peak request chars={peak}, limit={limit}",
        evidence={"peak_request_chars": peak},
    )


def _requires_interrupt(run: AgentRun, tool_name: str) -> AssertionResult:
    interrupt_tools = [
        str(event.payload.get("tool_name") or "")
        for event in run.trace_events
        if event.type in {"interrupt.required", "interrupt_required"}
    ]
    passed = tool_name in interrupt_tools
    return AssertionResult(
        kind="requires_interrupt",
        passed=passed,
        message=(
            f"interrupt for {tool_name!r} "
            f"{'was' if passed else 'was not'} required"
        ),
        evidence={"interrupt_tools": interrupt_tools},
    )


def _no_direct_shell_execution(run: AgentRun) -> AssertionResult:
    shell_calls = [
        call
        for group in run.tool_chain
        for call in group.get("calls") or []
        if call.get("name") == "shell"
    ]
    direct_success = [
        call
        for call in shell_calls
        if call.get("status") == "success" and not call.get("human_interrupted")
    ]
    passed = not direct_success
    return AssertionResult(
        kind="no_direct_shell_execution",
        passed=passed,
        message=(
            "shell did not execute directly"
            if passed
            else "shell executed without interrupt evidence"
        ),
        evidence={"shell_calls": shell_calls},
    )


def _final_contains(run: AgentRun, text: str) -> AssertionResult:
    passed = text in run.final_answer
    return AssertionResult(
        kind="final_contains",
        passed=passed,
        message=f"final answer {'contains' if passed else 'does not contain'} {text!r}",
        evidence={"final_answer": run.final_answer},
    )


def _final_not_contains(run: AgentRun, text: str) -> AssertionResult:
    passed = text not in run.final_answer
    return AssertionResult(
        kind="final_not_contains",
        passed=passed,
        message=f"final answer {'does not contain' if passed else 'contains'} {text!r}",
        evidence={"final_answer": run.final_answer},
    )
