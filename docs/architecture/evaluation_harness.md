# Agent Evaluation & Constraint Harness

The evaluation harness measures agent behavior, not only final answers. It is designed to compare the native backend and the optional LangGraph backend with the same scenario definitions.

## What It Evaluates

- Tool-use behavior: whether the agent selects the expected tool.
- Safety and permissions: whether high-risk tools require an interrupt.
- Cost: prompt chars, messages JSON chars, tool schema chars, and estimated input tokens.
- Trace quality: LLM calls, tool calls, tool results, interrupts, checkpoints, and final answers.
- Backend consistency: native and LangGraph runs can be compared without requiring identical natural-language wording.

## Core Modules

- `evaluation_system/harness/scenario.py`: scenario, config, run, trace, assertion, and metric data models.
- `evaluation_system/harness/runner.py`: executes a scenario against a selected backend.
- `evaluation_system/harness/recorder.py`: records trace events for one agent run.
- `evaluation_system/harness/report.py`: writes JSON and Markdown run reports.
- `evaluation_system/harness/compare.py`: runs native / LangGraph backend comparison.
- `evaluation_system/harness/cost.py`: captures prompt and tool schema cost without sending a real API request.

## Offline Mode

Harness tests use a fake provider and dummy tools. This keeps behavior checks reproducible and avoids spending real API tokens during normal testing.

## Useful Commands

```bash
uv run python3 -m pytest tests/test_evaluation_harness.py -v
uv run python3 -m pytest tests/test_evaluation_harness.py tests/test_langgraph_runtime_config.py tests/test_langgraph_runtime_reasoner.py -v -rs
```
