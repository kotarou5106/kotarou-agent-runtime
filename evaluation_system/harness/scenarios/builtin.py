from __future__ import annotations

from agent_runtime.provider import LLMResponse, ToolCall

from evaluation_system.harness.dummy_tools import DummyTool
from evaluation_system.harness.scenario import AssertionSpec, Scenario


def safe_tool_call() -> Scenario:
    return Scenario(
        id="safe_tool_call",
        description="普通安全工具调用成功，并返回最终答案。",
        input_messages=[{"role": "user", "content": "call the safe tool"}],
        provider_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_safe_1",
                        name="dummy_safe_tool",
                        arguments={"value": "ok"},
                    )
                ],
            ),
            LLMResponse(content="safe tool done", tool_calls=[]),
        ],
        tools=[DummyTool("dummy_safe_tool", result="dummy-result")],
        assertions=[
            AssertionSpec("required_tool", "dummy_safe_tool"),
            AssertionSpec("max_tool_calls", 1),
            AssertionSpec("final_contains", "done"),
        ],
    )


def shell_requires_interrupt() -> Scenario:
    return Scenario(
        id="shell_requires_interrupt",
        description="LangGraph 高风险 shell 工具必须被风险门标记为需要 interrupt。",
        input_messages=[{"role": "user", "content": "run shell"}],
        provider_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_shell_1",
                        name="shell",
                        arguments={"command": "echo hello"},
                    )
                ],
            ),
            LLMResponse(content="shell was not executed directly", tool_calls=[]),
        ],
        tools=[DummyTool("shell", result="shell-result")],
        tool_risks={"shell": "external-side-effect"},
        assertions=[
            AssertionSpec("requires_interrupt", "shell"),
            AssertionSpec("no_direct_shell_execution"),
        ],
        backends=("langgraph",),
    )


def langgraph_tool_loop() -> Scenario:
    scenario = safe_tool_call()
    scenario.id = "langgraph_tool_loop"
    scenario.description = "LangGraph LLM -> Tool -> LLM -> Final 工具循环。"
    scenario.backends = ("langgraph",)
    return scenario


def safe_tool_call_report() -> Scenario:
    scenario = safe_tool_call()
    scenario.id = "safe_tool_call_report"
    scenario.description = "生成 JSON / Markdown report 的安全工具调用场景。"
    return scenario


def backend_consistency_tool_loop() -> Scenario:
    scenario = safe_tool_call()
    scenario.id = "backend_consistency_tool_loop"
    scenario.description = "native 和 langgraph 都应跑通同一个 fake tool loop。"
    return scenario


def token_budget_comparison() -> Scenario:
    scenario = safe_tool_call()
    scenario.id = "token_budget_comparison"
    scenario.description = "比较 native/langgraph 的离线请求字符预算。"
    return scenario
