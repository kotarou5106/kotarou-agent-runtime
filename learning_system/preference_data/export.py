from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning_system.preference_data.builders import (
    EvaluationHarnessPreferenceBuilder,
    LLMJudgePreferenceBuilder,
    UserFeedbackPreferenceBuilder,
)
from learning_system.preference_data.io import (
    iter_input_records,
    write_preference_jsonl,
    write_trl_jsonl,
)
from learning_system.preference_data.schema import PreferenceSample
from learning_system.preference_data.synthetic_preference_builder import build_synthetic_samples


def build_samples_from_input(
    input_path: Path | None,
    *,
    source: str = "auto",
    min_score_gap: float = 0.2,
    synthetic_limit: int = 8,
) -> list[PreferenceSample]:
    records = iter_input_records(input_path) if input_path is not None else []
    samples: list[PreferenceSample] = _load_existing_preference_samples(
        records,
        min_score_gap=min_score_gap,
    )
    source = source.strip().lower()

    if samples and source == "auto":
        return _dedupe_samples(samples)

    if source in {"auto", "eval_harness"}:
        samples.extend(
            EvaluationHarnessPreferenceBuilder().build(
                records,
                min_score_gap=min_score_gap,
            )
        )
    if source in {"auto", "llm_judge"}:
        samples.extend(
            LLMJudgePreferenceBuilder().build(
                records,
                min_score_gap=min_score_gap,
            )
        )
    if source in {"auto", "user_feedback"}:
        feedback_records = [
            item for item in records if item.get("feedback_id") and item.get("feedback")
        ]
        if feedback_records:
            samples.extend(
                UserFeedbackPreferenceBuilder().build(
                    feedback_records,
                    min_score_gap=min_score_gap,
                )
            )
    if source in {"synthetic_pair", "synthetic"} or (source == "auto" and not samples):
        samples.extend(
            build_synthetic_samples(limit=synthetic_limit, min_score_gap=min_score_gap)
        )

    return _dedupe_samples(samples)


def _load_existing_preference_samples(
    records: list[dict],
    *,
    min_score_gap: float,
) -> list[PreferenceSample]:
    samples: list[PreferenceSample] = []
    for record in records:
        if not {
            "sample_id",
            "prompt",
            "chosen",
            "rejected",
            "source",
            "task_type",
            "judge_score_chosen",
            "judge_score_rejected",
            "score_gap",
        }.issubset(record):
            continue
        sample = PreferenceSample.model_validate(record)
        if sample.score_gap >= min_score_gap:
            samples.append(sample)
    return samples


def export_preferences(
    *,
    input_path: Path | None,
    output_path: Path,
    full_output_path: Path | None = None,
    source: str = "auto",
    min_score_gap: float = 0.2,
    synthetic_limit: int = 8,
) -> dict[str, object]:
    samples = build_samples_from_input(
        input_path,
        source=source,
        min_score_gap=min_score_gap,
        synthetic_limit=synthetic_limit,
    )
    full_path = full_output_path or output_path.with_name(
        output_path.stem + ".full" + output_path.suffix
    )
    write_trl_jsonl(samples, output_path)
    write_preference_jsonl(samples, full_path)
    return {
        "count": len(samples),
        "output": str(output_path),
        "full_output": str(full_path),
        "source": source,
        "min_score_gap": min_score_gap,
    }


def _dedupe_samples(samples: list[PreferenceSample]) -> list[PreferenceSample]:
    seen: set[str] = set()
    result: list[PreferenceSample] = []
    for sample in samples:
        if sample.sample_id in seen:
            continue
        seen.add(sample.sample_id)
        result.append(sample)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export DPO preference JSONL.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=["auto", "eval_harness", "llm_judge", "user_feedback", "synthetic_pair", "synthetic"],
        default="auto",
    )
    parser.add_argument("--min-score-gap", type=float, default=0.2)
    parser.add_argument("--synthetic-limit", type=int, default=8)
    args = parser.parse_args(argv)

    summary = export_preferences(
        input_path=args.input,
        output_path=args.output,
        full_output_path=args.full_output,
        source=args.source,
        min_score_gap=args.min_score_gap,
        synthetic_limit=args.synthetic_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
