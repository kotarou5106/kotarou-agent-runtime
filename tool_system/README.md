# tool_system

Tool use, tool discovery, execution hooks, and MCP tool access.

- `registry/`: tool catalog and schemas.
- `execution/`: running tool calls and writing tool messages.
- `built_in_tools/`: built-in tools exposed to the model.
- `tool_discovery/`: selecting deferred tools with tool_search.
- `mcp_tools/`: MCP client and MCP-backed tool registration.
- `tool_hooks/`: pre/post tool interception.

This maps to Tool Use, MCP, Exploration and Discovery, Guardrails, and Recovery.
