# agent_runtime

How one Agent turn runs.

This folder is for the core runtime patterns:

- lifecycle phases
- prompt construction
- reasoning loop
- skills loaded into context
- turn loop and interrupt state

Current implementation lives in `agent_runtime/looping`,
`agent_runtime/lifecycle`, `agent_runtime/context.py`, and
`agent_runtime/core/prompt_block.py`.
