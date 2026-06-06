from __future__ import annotations

import json

from learning_system.dpo.evaluate_dpo import evaluate_models
from learning_system.dpo.evaluate_dpo import main as eval_main
from learning_system.dpo.train_dpo import main as train_main, run_dpo_training
from learning_system.dpo.config import DPOTrainConfig
from learning_system.preference_data.io import read_jsonl


def test_train_dpo_dry_run_validates_dataset_without_gpu(tmp_path) -> None:
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Q", "chosen": "A good answer", "rejected": "Bad"}) + "\n",
        encoding="utf-8",
    )

    summary = run_dpo_training(
        DPOTrainConfig(
            dataset_path=str(dataset),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
        )
    )

    assert summary["dry_run"] is True
    assert summary["num_samples"] == 1


def test_train_dpo_cli_dry_run(tmp_path) -> None:
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Q", "chosen": "Chosen", "rejected": "Rejected"}) + "\n",
        encoding="utf-8",
    )

    assert train_main(["--dataset-path", str(dataset), "--dry-run"]) == 0


def test_evaluate_dpo_dry_run_writes_metrics(tmp_path) -> None:
    prompts = tmp_path / "eval.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Explain memory retrieval", "reference": "memory retrieval"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "eval_result.json"

    result = evaluate_models(
        base_model="base",
        dpo_model="dpo",
        prompts_path=prompts,
        output_path=output,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["num_prompts"] == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert "win_rate" in saved
    assert saved["items"][0]["base_answer"]
    assert saved["items"][0]["dpo_answer"]


def test_evaluate_dpo_dry_run_uses_builtin_prompts_when_file_missing(tmp_path) -> None:
    missing_prompts = tmp_path / "missing" / "eval_prompts.jsonl"
    output = tmp_path / "eval_result.json"

    result = evaluate_models(
        base_model="base",
        dpo_model="dpo",
        prompts_path=missing_prompts,
        output_path=output,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["num_prompts"] >= 1
    for key in ("win_rate", "avg_score_base", "avg_score_dpo", "score_delta"):
        assert key in result
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["items"][0]["metadata"]["source"] == "builtin_dry_run"


def test_evaluate_dpo_cli_dry_run_uses_builtin_prompts_when_file_missing(tmp_path) -> None:
    missing_prompts = tmp_path / "not-there.jsonl"
    output = tmp_path / "cli_eval_result.json"

    assert eval_main(
        [
            "--base-model",
            "base",
            "--dpo-model",
            "dpo",
            "--prompts",
            str(missing_prompts),
            "--output",
            str(output),
            "--dry-run",
        ]
    ) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["num_prompts"] >= 1
    for key in ("win_rate", "avg_score_base", "avg_score_dpo", "score_delta"):
        assert key in saved


def test_read_trl_jsonl_shape_for_training(tmp_path) -> None:
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Q", "chosen": "C", "rejected": "R"}) + "\n",
        encoding="utf-8",
    )

    assert set(read_jsonl(dataset)[0]) == {"prompt", "chosen", "rejected"}
