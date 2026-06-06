from __future__ import annotations

from typing import Any, cast

from agent_runtime.core.passive_turn import DefaultReasoner
from agent_runtime.core.runtime_support import LLMServices, ToolDiscoveryState
from agent_runtime.langgraph_runtime import LangGraphReasoner
from agent_runtime.langgraph_runtime.interrupts import ToolInterruptPolicy
from agent_runtime.looping.ports import LLMConfig
from agent_runtime.tools.registry import ToolRegistry

from evaluation_system.harness.assertions import evaluate_assertions
from evaluation_system.harness.fake_provider import ScriptedProvider
from evaluation_system.harness.recorder import build_run_trace_events
from evaluation_system.harness.scenario import (
    AgentRun,
    HarnessConfig,
    Scenario,
    TraceEvent,
)


class HarnessRunner:
    """Offline runner that reuses the project's native/LangGraph reasoners."""

    async def run(self, scenario: Scenario, config: HarnessConfig) -> AgentRun:
        provider = ScriptedProvider(scenario.provider_responses)
        registry = self._build_registry(scenario)
        trace_events, gated_tools = self._preflight_interrupts(
            scenario,
            config,
            registry,
        )
        reasoner = self._build_reasoner(provider, registry, config)
        disabled_tools = set(config.disabled_tools) | gated_tools
        final_answer = ""
        tool_chain: list[dict[str, Any]] = []
        tools_used: list[str] = []
        metadata: dict[str, Any] = {}
        error: str | None = None
        try:
            result = await reasoner.run(
                list(scenario.input_messages),
                disabled_tools=disabled_tools,
                tool_event_session_key=f"eval:{scenario.id}:{config.backend}",
                tool_event_channel="eval",
                tool_event_chat_id=scenario.id,
                trace_id=f"eval:{scenario.id}:{config.backend}",
            )
            final_answer = result.reply
            tool_chain = list(result.metadata.get("tool_chain") or [])
            tools_used = list(result.metadata.get("tools_used") or [])
            metadata = dict(result.metadata)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        trace_events = build_run_trace_events(
            scenario_id=scenario.id,
            backend=config.backend,
            input_messages=list(scenario.input_messages),
            llm_calls=list(provider.calls),
            tool_chain=tool_chain,
            preflight_events=trace_events,
            final_answer=final_answer,
            passed=error is None,
            error=error,
            checkpoint_enabled=config.checkpoint_enabled,
        )
        run = AgentRun(
            scenario_id=scenario.id,
            backend=config.backend,
            final_answer=final_answer,
            trace_events=trace_events,
            tool_chain=tool_chain,
            tools_used=tools_used,
            llm_calls=list(provider.calls),
            error=error,
            metadata=metadata,
        )
        run.assertion_results = evaluate_assertions(run, scenario.assertions)
        for event in reversed(run.trace_events):
            if event.type == "run_finished":
                event.metadata["passed"] = run.passed
                event.payload["passed"] = run.passed
                break
        return run

    def _build_registry(self, scenario: Scenario) -> ToolRegistry:
        registry = ToolRegistry()
        always = (
            set(scenario.always_on_tools)
            if scenario.always_on_tools is not None
            else {tool.name for tool in scenario.tools}
        )
        for tool in scenario.tools:
            registry.register(
                tool,
                always_on=tool.name in always,
                risk=scenario.tool_risks.get(tool.name, "read-only"),
            )
        return registry

    def _build_reasoner(
        self,
        provider: ScriptedProvider,
        registry: ToolRegistry,
        config: HarnessConfig,
    ) -> DefaultReasoner:
        llm_config = LLMConfig(
            model=config.model,
            max_iterations=config.max_iterations,
            max_tokens=config.max_tokens,
            tool_search_enabled=config.tool_search_enabled,
        )
        common: dict[str, Any] = {
            "llm": cast(
                Any,
                LLMServices(
                    provider=cast(Any, provider),
                    light_provider=cast(Any, provider),
                ),
            ),
            "llm_config": llm_config,
            "tools": registry,
            "discovery": ToolDiscoveryState(),
            "tool_search_enabled": config.tool_search_enabled,
            "memory_window": 40,
        }
        if config.backend == "langgraph":
            return LangGraphReasoner(
                **common,
                interrupt_policy=ToolInterruptPolicy(
                    enabled=False,
                ),
                checkpoint_persistent=config.checkpoint_enabled,
            )
        return DefaultReasoner(**common)

    def _preflight_interrupts(
        self,
        scenario: Scenario,
        config: HarnessConfig,
        registry: ToolRegistry,
    ) -> tuple[list[TraceEvent], set[str]]:
        if config.backend != "langgraph" or not config.interrupt_high_risk_tools:
            return [], set()
        policy = ToolInterruptPolicy(enabled=True)
        events: list[TraceEvent] = []
        gated_tools: set[str] = set()
        for response in scenario.provider_responses:
            for call in response.tool_calls:
                if policy.requires_approval(registry, call):
                    gated_tools.add(call.name)
                    events.append(
                        TraceEvent(
                            "interrupt.required",
                            {
                                "tool_name": call.name,
                                "call_id": call.id,
                                "arguments": dict(call.arguments),
                            },
                        )
                    )
        return events, gated_tools
