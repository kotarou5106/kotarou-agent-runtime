from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from learning_system.preference_data.builders import score_text_against_reference
from learning_system.preference_data.io import read_json_or_jsonl


@dataclass(frozen=True)
class EvalPrompt:
    prompt: str
    reference: str = ""
    task_type: str = "general_chat"
    metadata: dict | None = None


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockGenerator:
    def __init__(self, label: str) -> None:
        self.label = label

    def generate(self, prompt: str) -> str:
        prefix = "Improved answer" if self.label == "dpo" else "Baseline answer"
        return f"{prefix}: {prompt.strip()}"


class TransformersGenerator:
    def __init__(self, model_name: str, *, max_new_tokens: int = 128) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    def generate(self, prompt: str) -> str:
        result = self._pipe(prompt)
        if isinstance(result, list) and result:
            text = str(result[0].get("generated_text") or "")
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        return ""


def builtin_mock_eval_prompts() -> list[EvalPrompt]:
    return [
        EvalPrompt(
            prompt="Explain how the agent should use long-term memory when answering a user.",
            reference="agent long-term memory answer user",
            task_type="memory_qa",
            metadata={"source": "builtin_dry_run"},
        ),
        EvalPrompt(
            prompt="Summarize the tool-use trace and identify whether the final answer is grounded.",
            reference="tool trace final answer grounded",
            task_type="tool_use",
            metadata={"source": "builtin_dry_run"},
        ),
        EvalPrompt(
            prompt="Answer a RAG question using retrieved context and citations.",
            reference="rag retrieved context citations",
            task_type="rag_qa",
            metadata={"source": "builtin_dry_run"},
        ),
    ]


def load_eval_prompts(path: Path, *, allow_builtin: bool = False) -> list[EvalPrompt]:
    if allow_builtin and not path.exists():
        return builtin_mock_eval_prompts()
    records = read_json_or_jsonl(path)
    if allow_builtin and not records:
        return builtin_mock_eval_prompts()
    prompts: list[EvalPrompt] = []
    for index, record in enumerate(records, start=1):
        prompt = str(record.get("prompt") or record.get("question") or "").strip()
        if not prompt:
            raise ValueError(f"missing prompt at eval record {index}")
        prompts.append(
            EvalPrompt(
                prompt=prompt,
                reference=str(record.get("reference") or record.get("gold") or "").strip(),
                task_type=str(record.get("task_type") or "general_chat"),
                metadata=dict(record.get("metadata") or {}),
            )
        )
    if not prompts:
        if allow_builtin:
            return builtin_mock_eval_prompts()
        raise ValueError(f"no eval prompts found: {path}")
    return prompts


def evaluate_models(
    *,
    base_model: str,
    dpo_model: str,
    prompts_path: Path,
    output_path: Path,
    dry_run: bool = False,
    max_new_tokens: int = 128,
) -> dict:
    prompts = load_eval_prompts(prompts_path, allow_builtin=dry_run)
    if dry_run:
        base_generator: TextGenerator = MockGenerator("base")
        dpo_generator: TextGenerator = MockGenerator("dpo")
    else:
        base_generator = TransformersGenerator(base_model, max_new_tokens=max_new_tokens)
        dpo_generator = TransformersGenerator(dpo_model, max_new_tokens=max_new_tokens)

    items: list[dict] = []
    base_scores: list[float] = []
    dpo_scores: list[float] = []
    dpo_wins = 0
    ties = 0
    for item in prompts:
        base_answer = base_generator.generate(item.prompt)
        dpo_answer = dpo_generator.generate(item.prompt)
        base_score = _score_answer(base_answer, item)
        dpo_score = _score_answer(dpo_answer, item)
        base_scores.append(base_score)
        dpo_scores.append(dpo_score)
        if dpo_score > base_score:
            dpo_wins += 1
        elif dpo_score == base_score:
            ties += 1
        items.append(
            {
                "prompt": item.prompt,
                "task_type": item.task_type,
                "reference": item.reference,
                "base_answer": base_answer,
                "dpo_answer": dpo_answer,
                "base_score": base_score,
                "dpo_score": dpo_score,
                "winner": "dpo" if dpo_score > base_score else "base" if base_score > dpo_score else "tie",
                "metadata": item.metadata or {},
            }
        )

    n = max(1, len(items))
    result = {
        "dry_run": dry_run,
        "base_model": base_model,
        "dpo_model": dpo_model,
        "num_prompts": len(items),
        "win_rate": round(dpo_wins / n, 4),
        "tie_rate": round(ties / n, 4),
        "avg_score_base": round(sum(base_scores) / n, 4),
        "avg_score_dpo": round(sum(dpo_scores) / n, 4),
        "score_delta": round((sum(dpo_scores) - sum(base_scores)) / n, 4),
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))
    return result


def _score_answer(answer: str, prompt: EvalPrompt) -> float:
    if prompt.reference:
        return round(score_text_against_reference(answer, prompt.reference), 4)
    text = answer.strip()
    if not text:
        return 0.0
    length_score = min(1.0, len(text) / 200.0)
    prompt_terms = {token.lower() for token in prompt.prompt.split() if len(token) > 3}
    answer_terms = {token.lower() for token in text.split()}
    overlap = len(prompt_terms & answer_terms) / max(1, len(prompt_terms))
    return round(0.5 * length_score + 0.5 * overlap, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate base vs DPO model responses.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dpo-model", default="outputs/dpo")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    evaluate_models(
        base_model=args.base_model,
        dpo_model=args.dpo_model,
        prompts_path=args.prompts,
        output_path=args.output,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
