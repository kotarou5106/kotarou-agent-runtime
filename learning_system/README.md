# Learning System

`learning_system/` contains offline post-training utilities for the Personal AI
Agent Runtime. It does not train inside the live agent loop.

## DPO Preference Optimization

The DPO pipeline is:

1. Agent Runtime / RAG / tool / eval / feedback logs
2. Preference dataset builders
3. JSONL export with `prompt`, `chosen`, `rejected`
4. Offline TRL `DPOTrainer`
5. Before/after evaluation
6. Optional deployment of the trained model back into runtime config

Supported preference sources:

- Eval harness reports and runs from `evaluation_system/harness/runner.py`.
- LongMemEval LLM-as-a-judge scores from `evaluation_system/longmemeval/metrics.py`.
- Knowledge retrieval/RAG traces from `knowledge_system/retrieval/retriever.py`.
- Tool traces from `agent_runtime/core/runtime_support.py` and harness reports.
- Proactive scoring from `proactive_system/judge.py`.
- Explicit user feedback via `UserFeedbackEvent` in
  `learning_system/preference_data/schema.py`.

Dry-run export:

```bash
python -m learning_system.preference_data.export \
  --input data/agent_logs \
  --output data/dpo/preferences.jsonl \
  --min-score-gap 0.2
```

If no usable records are found in `--source auto`, the exporter emits a small
`source="synthetic_pair"` dataset for smoke testing. Synthetic samples are not
real user feedback and should not be used to claim production preference
learning.

Dry-run training validation:

```bash
python -m learning_system.dpo.train_dpo \
  --dataset-path data/dpo/preferences.jsonl \
  --output-dir outputs/dpo \
  --dry-run
```

Real training requires the optional DPO dependencies and model downloads:

```bash
pip install -r requirements-dpo.txt
python -m learning_system.dpo.train_dpo \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-path data/dpo/preferences.jsonl \
  --output-dir outputs/dpo/qwen-agent-dpo \
  --use-lora
```

Dry-run evaluation:

```bash
python -m learning_system.dpo.evaluate_dpo \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --dpo-model outputs/dpo/qwen-agent-dpo \
  --prompts data/dpo/eval_prompts.jsonl \
  --output outputs/dpo/eval_result.json \
  --dry-run
```

