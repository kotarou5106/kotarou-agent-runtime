# Personal Agent OS Directory Structure

This repository is organized as a Personal AI Agent OS. The top-level folders
describe Agent capabilities directly, so a reader can understand the system by
following what the Agent can do.

## Capability Map

| Folder | Agent pattern focus | Responsibility |
| --- | --- | --- |
| `app/` | App startup, human entrypoints | CLI entrypoints, startup adapters, app-level routing |
| `agent_runtime/` | Prompt chaining, reasoning, tool loop | turn lifecycle, prompt assembly, LLM loop, tools, MCP client, built-in skills |
| `conversation_patterns/` | Routing, passive chat, proactive chat | reusable conversation flow patterns and response parsing |
| `memory_system/` | Memory management, RAG, correction | long-term memory APIs, semantic retrieval, consolidation, correction |
| `tool_system/` | Tool use, discovery, execution | tool registry, tool execution, hooks, MCP-backed tools |
| `planning_system/` | Planning, background work | scheduler, drift tasks, resource-aware maintenance |
| `proactive_system/` | Goal monitoring, priority ranking | proactive tick, sources, judge, dedupe, delivery, ACK |
| `multi_agent_system/` | Multi-agent collaboration, A2A | local subagents, peer agents, collaboration adapters |
| `connectors/` | External integrations | channels, persistence helpers, HTTP/network helpers, MCP servers |
| `safety_system/` | Guardrails and recovery | shell safety, loop guard, undo, recovery patterns |
| `evaluation_system/` | Evaluation and monitoring | memory benchmarks, benchmark runtime, result analysis |
| `dashboard/` | Human-in-the-loop observability | FastAPI backend, React frontend, static assets, plugin panels |
| `storage/` | Local state and persistence | sessions, message history, local stores |
| `docs/` | Documentation | architecture notes and handbooks |
| `scripts/` | Operations | maintenance scripts, Docker scripts, debug utilities |
| `tests/` | Verification | unit tests, smoke tests, behavior baselines |

## How To Read The Project

Start from `app/` for entrypoints, then follow the runtime into
`agent_runtime/`. Memory-related behavior lives in `memory_system/`; proactive
behavior lives in `proactive_system/`; external systems live in `connectors/`;
human inspection lives in `dashboard/`.

The smaller capability folders such as `conversation_patterns/`,
`planning_system/`, `tool_system/`, `multi_agent_system/`, and `safety_system/`
act as a learning map for common Agent design patterns.
