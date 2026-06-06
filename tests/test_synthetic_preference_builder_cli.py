from __future__ import annotations

import json
import subprocess
import sys

from learning_system.preference_data.io import read_jsonl
from learning_system.preference_data.schema import PreferenceSample


def test_synthetic_preference_builder_cli_creates_full_metadata_jsonl(tmp_path) -> None:
    output = tmp_path / "missing" / "nested" / "synthetic_preferences.full.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "learning_system.preference_data.synthetic_preference_builder",
            "--output",
            str(output),
            "--limit",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["count"] >= 1
    assert summary["output"] == str(output)
    assert output.exists()

    records = read_jsonl(output)
    assert len(records) == summary["count"]
    sample = PreferenceSample.model_validate(records[0])
    assert sample.source == "synthetic_pair"
    assert sample.prompt
    assert sample.chosen
    assert sample.rejected
    assert sample.chosen != sample.rejected
    assert sample.score_gap >= 0.2
    assert sample.metadata["note"].startswith("Synthetic pair")


def test_export_cli_converts_full_synthetic_jsonl_to_trl_jsonl(tmp_path) -> None:
    full_output = tmp_path / "dpo" / "synthetic_preferences.full.jsonl"
    trl_output = tmp_path / "dpo" / "preferences.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "learning_system.preference_data.synthetic_preference_builder",
            "--output",
            str(full_output),
            "--limit",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "learning_system.preference_data.export",
            "--input",
            str(full_output),
            "--output",
            str(trl_output),
            "--min-score-gap",
            "0.2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["count"] >= 1
    assert trl_output.exists()
    records = read_jsonl(trl_output)
    assert set(records[0]) == {"prompt", "chosen", "rejected"}
    assert records[0]["prompt"]
    assert records[0]["chosen"]
    assert records[0]["rejected"]

