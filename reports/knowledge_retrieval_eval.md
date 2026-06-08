# Knowledge Retrieval Evaluation Report

## Summary

| Retriever | Recall@1 | Recall@3 | Recall@5 | Precision@5 | HitRate@5 | MRR@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-case Results

### bm25

- `agent-runtime-loop` miss
  - query: How does the Agent Runtime main loop assemble context and call the LLM?
  - relevant ids: agent_runtime
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_cfc593ad79f64f26806c4ee9
- `long-term-memory` miss
  - query: How is long-term memory retrieved into the current conversation?
  - relevant ids: memory_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_66ccafee779b4a95a9747a0d
- `tool-calling` miss
  - query: Where are tool calling and function calling handled?
  - relevant ids: tool_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_daa2a026e4254b6a9dce51ec
- `plugin-system` miss
  - query: How can plugins register tools or lifecycle hooks?
  - relevant ids: plugins
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_68a0dd18072f41159fa016fd
- `proactive-scheduler` miss
  - query: How do proactive tasks and scheduler background jobs run?
  - relevant ids: proactive_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_e97349c565fa4dc29f374c20
- `langgraph-workflow` miss
  - query: What does the LangGraph workflow orchestration backend do?
  - relevant ids: langgraph_workflow
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_d61e1ff544ed4eb89a0164de
- `hybrid-rag` miss
  - query: How does Hybrid RAG use vector search BM25 and RRF?
  - relevant ids: knowledge_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_7fa6fbed51024a7ba49c14cc
- `citation-validation` miss
  - query: How are knowledge citations validated?
  - relevant ids: citation
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_70391fd09025446c9ecebbf5
- `dashboard-observability` miss
  - query: How can the dashboard inspect knowledge retrieval traces?
  - relevant ids: dashboard
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_01aeca0d24f74a3d807ba3ce
- `token-cost-evaluation` miss
  - query: How does token cost evaluation estimate prompt and tool schema cost?
  - relevant ids: token_cost_optimization
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_01ef4fe88f004ceab34ae62e

### vector

- `agent-runtime-loop` miss
  - query: How does the Agent Runtime main loop assemble context and call the LLM?
  - relevant ids: agent_runtime
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_86ff19f7894d4bd1b49e15e6
- `long-term-memory` miss
  - query: How is long-term memory retrieved into the current conversation?
  - relevant ids: memory_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_5d88e5508fff415fb9d11819
- `tool-calling` miss
  - query: Where are tool calling and function calling handled?
  - relevant ids: tool_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_d222d19977dd4604b94062b9
- `plugin-system` miss
  - query: How can plugins register tools or lifecycle hooks?
  - relevant ids: plugins
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_b2123364a32c45138d4879e4
- `proactive-scheduler` miss
  - query: How do proactive tasks and scheduler background jobs run?
  - relevant ids: proactive_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_6bbf2a0a9b0f42238004e0ac
- `langgraph-workflow` miss
  - query: What does the LangGraph workflow orchestration backend do?
  - relevant ids: langgraph_workflow
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_e028af9fad174fd8a1f7277b
- `hybrid-rag` miss
  - query: How does Hybrid RAG use vector search BM25 and RRF?
  - relevant ids: knowledge_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_91ad5799929e450c81d1ab44
- `citation-validation` miss
  - query: How are knowledge citations validated?
  - relevant ids: citation
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_eff3f1d105ad4815a8a1e313
- `dashboard-observability` miss
  - query: How can the dashboard inspect knowledge retrieval traces?
  - relevant ids: dashboard
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_39aa60a4c4b74c9eb5163cf4
- `token-cost-evaluation` miss
  - query: How does token cost evaluation estimate prompt and tool schema cost?
  - relevant ids: token_cost_optimization
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_797bed3e822b482e8e665971

### hybrid_rrf

- `agent-runtime-loop` miss
  - query: How does the Agent Runtime main loop assemble context and call the LLM?
  - relevant ids: agent_runtime
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_bab00a2c88a14c51b0af6afd
- `long-term-memory` miss
  - query: How is long-term memory retrieved into the current conversation?
  - relevant ids: memory_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_e89ed989b8414f22a7563688
- `tool-calling` miss
  - query: Where are tool calling and function calling handled?
  - relevant ids: tool_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_7aa80874e00c4cc3ad45ffd4
- `plugin-system` miss
  - query: How can plugins register tools or lifecycle hooks?
  - relevant ids: plugins
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_f59048823b9e4fee94d1fd18
- `proactive-scheduler` miss
  - query: How do proactive tasks and scheduler background jobs run?
  - relevant ids: proactive_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_4f426b64b0044ce4aaf0b6d2
- `langgraph-workflow` miss
  - query: What does the LangGraph workflow orchestration backend do?
  - relevant ids: langgraph_workflow
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_ffc2ae7c814a4866be18f392
- `hybrid-rag` miss
  - query: How does Hybrid RAG use vector search BM25 and RRF?
  - relevant ids: knowledge_system
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_7cbd30454aa94ec5a61d5b16
- `citation-validation` miss
  - query: How are knowledge citations validated?
  - relevant ids: citation
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_344983c28e70472eaa79ace8
- `dashboard-observability` miss
  - query: How can the dashboard inspect knowledge retrieval traces?
  - relevant ids: dashboard
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_b4c648ce6f1c437896b29b53
- `token-cost-evaluation` miss
  - query: How does token cost evaluation estimate prompt and tool schema cost?
  - relevant ids: token_cost_optimization
  - retrieved ids: -
  - first relevant rank: -
  - trace_id: ktrace_6ee35e97096d4f9fa1359593

## Notes

- This is a lightweight internal retrieval evaluation.
- It evaluates retrieval quality only, not final answer generation.
- Answer faithfulness and hallucination evaluation can be added later.
