from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning_system.preference_data.builders import infer_task_type, stable_sample_id
from learning_system.preference_data.io import write_preference_jsonl
from learning_system.preference_data.schema import PreferenceSample, PreferenceSource, TaskType


def build_synthetic_samples(*, limit: int = 8, min_score_gap: float = 0.2) -> list[PreferenceSample]:
    """Create traceable smoke-test DPO pairs from built-in eval scenarios.

    These samples are deliberately marked as synthetic_pair. They are useful for
    validating the DPO data/CLI/training plumbing, not for claiming real user
    preference learning.
    """
    from evaluation_system.harness.registry import list_scenarios, get_scenario

    samples: list[PreferenceSample] = []
    for name in list_scenarios():
        scenario = get_scenario(name)
        prompt = "\n".join(
            str(message.get("content") or "").strip()
            for message in scenario.input_messages
            if isinstance(message, dict) and message.get("content")
        ).strip()
        if not prompt:
            continue
        chosen = _last_text_response(scenario.provider_responses)
        if not chosen:
            continue
        rejected = "I cannot complete this task because I do not have enough context."
        if chosen.strip() == rejected:
            continue
        sample = PreferenceSample(
            sample_id=stable_sample_id("synthetic", scenario.id, chosen, rejected),
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source=PreferenceSource.SYNTHETIC_PAIR,
            task_type=infer_task_type({"scenario_id": scenario.id}) or TaskType.GENERAL_CHAT,
            judge_score_chosen=0.9,
            judge_score_rejected=0.1,
            score_gap=0.8,
            metadata={
                "scenario_id": scenario.id,
                "description": scenario.description,
                "note": "Synthetic pair for pipeline smoke tests; not real user feedback.",
            },
        )
        if sample.score_gap >= min_score_gap:
            samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def _last_text_response(responses: list[object]) -> str:
    for response in reversed(responses):
        content = str(getattr(response, "content", "") or "").strip()
        if content:
            return content
    return ""


def export_synthetic_preferences(
    *,
    output: Path,
    limit: int = 8,
    min_score_gap: float = 0.2,
) -> dict[str, object]:
    samples = build_synthetic_samples(limit=limit, min_score_gap=min_score_gap)
    write_preference_jsonl(samples, output)
    return {
        "count": len(samples),
        "output": str(output),
        "source": PreferenceSource.SYNTHETIC_PAIR.value,
        "min_score_gap": min_score_gap,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate full-metadata synthetic DPO PreferenceSample JSONL."
    )
    parser.add_argument("--output", "--output-path", "--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--min-score-gap", type=float, default=0.2)
    args = parser.parse_args(argv)

    summary = export_synthetic_preferences(
        output=args.output,
        limit=args.limit,
        min_score_gap=args.min_score_gap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
