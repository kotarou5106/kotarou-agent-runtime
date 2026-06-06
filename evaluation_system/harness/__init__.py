"""Agent behavior evaluation harness.

Phase 1 focuses on deterministic offline runs with fake providers and dummy
tools. It intentionally reuses the existing native and LangGraph reasoners
instead of introducing a separate agent loop.
"""

from evaluation_system.harness.assertions import evaluate_assertions
from evaluation_system.harness.compare import BackendComparisonRunner
from evaluation_system.harness.cost import CostProbe, CostRecorder, CostReport, PromptCostSnapshot
from evaluation_system.harness.report import Report
from evaluation_system.harness.runner import HarnessRunner
from evaluation_system.harness.scenario import (
    AgentRun,
    AssertionResult,
    ComparisonResult,
    HarnessConfig,
    Scenario,
    TraceEvent,
)

__all__ = [
    "AgentRun",
    "AssertionResult",
    "BackendComparisonRunner",
    "ComparisonResult",
    "CostProbe",
    "CostRecorder",
    "CostReport",
    "HarnessConfig",
    "HarnessRunner",
    "PromptCostSnapshot",
    "Report",
    "Scenario",
    "TraceEvent",
    "evaluate_assertions",
]
