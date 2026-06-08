# memory_system

Long-term memory, semantic retrieval, consolidation, and correction.

- `markdown_memory/`: MEMORY.md, SELF.md, HISTORY.md, PENDING.md, RECENT_CONTEXT.md.
- `semantic_memory/`: SQLite vector memory and memory item lifecycle.
- `retrieval/`: RAG pipelines used before reasoning.
- `consolidation/`: conversation-to-memory extraction.
- `correction/`: post-response correction hooks and memory invalidation helpers.
- `memory_tools/`: recall, memorize, forget tools.

Current implementation is in `agent_runtime/core/memory`,
`agent_runtime/memory.py`, `memory_system`, and `agent_runtime/retrieval`.
