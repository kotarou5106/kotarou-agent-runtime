from __future__ import annotations

import json

import pytest

from learning_system.preference_data.builders import LLMJudgePreferenceBuilder
from learning_system.preference_data.export import export_preferences
from learning_system.preference_data.io import read_jsonl
from learning_system.preference_data.schema import (
    PreferenceSample,
    PreferenceSource,
    TaskType,
    filter_samples,
)


def _sample(**overrides) -> PreferenceSample:
    data = {
        "sample_id": "s1",
        "prompt": "How should the agent answer?",
        "chosen": "Use the relevant memory and cite the source.",
        "rejected": "I do not know.",
        "source": PreferenceSource.LLM_JUDGE,
        "task_type": TaskType.MEMORY_QA,
        "judge_score_chosen": 0.9,
        "judge_score_rejected": 0.2,
        "score_gap": 0.7,
        "metadata": {"turn_id": "t1"},
    }
    data.update(overrides)
    return PreferenceSample.model_validate(data)


def test_preference_sample_schema_accepts_valid_pair() -> None:
    sample = _sample()

    assert sample.to_trl_record() == {
        "prompt": sample.prompt,
        "chosen": sample.chosen,
        "rejected": sample.rejected,
    }
    assert sample.metadata["turn_id"] == "t1"


def test_preference_sample_rejects_empty_chosen_or_rejected() -> None:
    with pytest.raises(ValueError):
        _sample(chosen="")

    with pytest.raises(ValueError):
        _sample(rejected=" ")


def test_preference_sample_rejects_identical_pair() -> None:
    with pytest.raises(ValueError):
        _sample(chosen="same", rejected="same")


def test_score_gap_filter_drops_small_gap() -> None:
    keep = _sample(sample_id="keep", judge_score_chosen=0.8, judge_score_rejected=0.2, score_gap=0.6)
    drop = _sample(sample_id="drop", judge_score_chosen=0.55, judge_score_rejected=0.45, score_gap=0.1)

    assert filter_samples([keep, drop], min_score_gap=0.2) == [keep]


def test_llm_judge_builder_chooses_higher_scored_response() -> None:
    samples = LLMJudgePreferenceBuilder().build(
        [
            {
                "prompt": "Q",
                "response_a": "weak",
                "response_b": "strong",
                "score_a": 0.1,
                "score_b": 0.8,
                "task_type": "rag_qa",
            }
        ],
        min_score_gap=0.2,
    )

    assert len(samples) == 1
    assert samples[0].chosen == "strong"
    assert samples[0].rejected == "weak"
    assert samples[0].source == PreferenceSource.LLM_JUDGE


def test_exporter_outputs_trl_and_full_jsonl(tmp_path) -> None:
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "prompt": "Q",
                "response_a": "good answer",
                "response_b": "bad answer",
                "score_a": 0.9,
                "score_b": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "preferences.jsonl"
    full_output = tmp_path / "preferences.full.jsonl"

    summary = export_preferences(
        input_path=input_path,
        output_path=output,
        full_output_path=full_output,
        source="llm_judge",
        min_score_gap=0.2,
    )

    assert summary["count"] == 1
    trl_records = read_jsonl(output)
    full_records = read_jsonl(full_output)
    assert set(trl_records[0]) == {"prompt", "chosen", "rejected"}
    assert full_records[0]["source"] == "llm_judge"


def test_synthetic_export_generates_legal_jsonl(tmp_path) -> None:
    output = tmp_path / "synthetic.jsonl"

    summary = export_preferences(
        input_path=None,
        output_path=output,
        source="synthetic_pair",
        synthetic_limit=2,
    )

    records = read_jsonl(output)
    assert summary["count"] >= 1
    assert {"prompt", "chosen", "rejected"} == set(records[0])

