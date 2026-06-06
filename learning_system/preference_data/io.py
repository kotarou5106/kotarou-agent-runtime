from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from learning_system.preference_data.schema import PreferenceSample


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(json.loads(text))
    return records


def read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "samples", "results", "runs", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def iter_input_records(path: Path) -> list[dict]:
    if path.is_file():
        return read_json_or_jsonl(path)
    if not path.exists():
        return []
    records: list[dict] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        records.extend(read_json_or_jsonl(file_path))
    return records


def write_preference_jsonl(samples: Iterable[PreferenceSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    return path


def write_trl_jsonl(samples: Iterable[PreferenceSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_trl_record(), ensure_ascii=False) + "\n")
    return path


def load_preference_samples(path: Path) -> list[PreferenceSample]:
    return [PreferenceSample.model_validate(item) for item in read_json_or_jsonl(path)]

