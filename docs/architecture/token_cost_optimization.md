# Token Cost Evaluation & Optimization

This project includes an offline cost probe for the agent request assembled before `provider.chat(...)`. It captures messages, tool schemas, model, max tokens, and derived cost fields without sending a real LLM request.

## Measured Fields

- `system_chars`
- `messages_chars`
- `messages_json_chars`
- `tools_schema_chars`
- `tool_count`
- `always_on_tool_count`
- `estimated_input_tokens`
- `max_tokens`
- `model`
- `backend`

## Current Optimization Result

| Item | Before | After |
| --- | ---: | ---: |
| system prompt | 11,782 chars | 5,173 chars |
| messages JSON | 12,934 chars | 5,590 chars |
| tools schema JSON | 16,635 chars | 3,316 chars |
| always-on tools | 19 | 6 |
| estimated input tokens | 9,856 | 2,968 |

## What Changed

- The system prompt keeps short core identity and behavior rules in the default path.
- Detailed memory correction, history search, spawn, and tool-use guidance are kept as shorter on-demand rules.
- The always-on tool set was reduced to common low-risk tools.
- Higher-cost or higher-risk tools such as shell, spawn, write/edit, message push, and message history search remain discoverable as deferred tools.
- `progress_description` is no longer injected into every tool schema and is not required for ordinary tools.

## Relevant Files

- `evaluation_system/harness/cost.py`
- `agent_runtime/prompts/agent.py`
- `agent_runtime/tools/registry.py`
- `agent_runtime/tools/meta/register.py`
- `bootstrap/toolsets/meta.py`
- `tests/test_evaluation_harness.py`
