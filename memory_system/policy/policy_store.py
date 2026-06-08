from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memory_system.policy.reward_signal import MemoryPolicyEvent


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class MemoryPolicyStats:
    memory_id: str
    retrieved_count: int = 0
    injected_count: int = 0
    used_count: int = 0
    corrected_count: int = 0
    superseded_count: int = 0
    deleted_count: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    reliability: float = 0.5
    usefulness: float = 0.0
    retrieval_boost: float = 0.0
    last_event_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPolicyStats":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_reward"] = round(float(self.total_reward), 6)
        data["avg_reward"] = round(float(self.avg_reward), 6)
        data["reliability"] = round(float(self.reliability), 6)
        data["usefulness"] = round(float(self.usefulness), 6)
        data["retrieval_boost"] = round(float(self.retrieval_boost), 6)
        return data


class MemoryPolicyStore:
    """JSON-backed memory policy stats store.

    The policy state is separate from memory_items so it can be reset without
    altering long-term memory content.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def get_memory_stats(self, memory_id: str) -> MemoryPolicyStats:
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id is required")
        with self._lock:
            data = self._read()
            raw = data.get(memory_id)
            if isinstance(raw, dict):
                return MemoryPolicyStats.from_dict(raw)
            return MemoryPolicyStats(memory_id=memory_id)

    def update(self, memory_id: str, event: MemoryPolicyEvent) -> MemoryPolicyStats:
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id is required")
        with self._lock:
            data = self._read()
            raw = data.get(memory_id) if isinstance(data.get(memory_id), dict) else {}
            stats = MemoryPolicyStats.from_dict(
                {"memory_id": memory_id, **dict(raw or {})}
            )
            _apply_event(stats, event)
            data[memory_id] = stats.to_dict()
            self._write(data)
            return stats

    def list_stats(self) -> list[MemoryPolicyStats]:
        with self._lock:
            rows = [
                MemoryPolicyStats.from_dict({"memory_id": memory_id, **payload})
                for memory_id, payload in self._read().items()
                if isinstance(payload, dict)
            ]
        rows.sort(key=lambda item: item.retrieval_boost, reverse=True)
        return rows

    def reset(self) -> None:
        with self._lock:
            self._write({})

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)


def _apply_event(stats: MemoryPolicyStats, event: MemoryPolicyEvent) -> None:
    event_type = event.event_type
    reward = event.resolved_reward()
    if event_type == "retrieved":
        stats.retrieved_count += 1
    elif event_type == "injected":
        stats.injected_count += 1
    elif event_type == "used":
        stats.used_count += 1
    elif event_type == "corrected":
        stats.corrected_count += 1
    elif event_type == "superseded":
        stats.superseded_count += 1
    elif event_type == "deleted":
        stats.deleted_count += 1
    if reward > 0 and event_type in {"user_positive", "used"}:
        stats.positive_feedback_count += 1
    if reward < 0 and event_type in {"user_negative", "corrected"}:
        stats.negative_feedback_count += 1

    event_count = (
        stats.retrieved_count
        + stats.injected_count
        + stats.used_count
        + stats.corrected_count
        + stats.superseded_count
        + stats.deleted_count
        + stats.positive_feedback_count
        + stats.negative_feedback_count
    )
    stats.total_reward += reward
    stats.avg_reward = stats.total_reward / max(1, event_count)
    stats.last_event_at = event.created_at
    _recompute_scores(stats)


def _recompute_scores(stats: MemoryPolicyStats) -> None:
    positive = stats.used_count + stats.positive_feedback_count
    negative = stats.corrected_count + stats.superseded_count + stats.deleted_count
    exposure = max(1, stats.retrieved_count + stats.injected_count)

    reliability = (
        0.50
        + 0.015 * min(stats.retrieved_count, 20)
        + 0.035 * min(stats.injected_count, 20)
        + 0.090 * min(positive, 10)
        - 0.180 * min(stats.corrected_count, 10)
        - 0.140 * min(stats.superseded_count, 10)
        - 0.180 * min(stats.deleted_count, 10)
        - 0.120 * min(stats.negative_feedback_count, 10)
    )
    usefulness = (
        0.50 * stats.injected_count
        + 1.50 * stats.used_count
        + 2.00 * stats.positive_feedback_count
    ) / max(1.0, float(exposure + positive + negative))
    boost = (
        0.24 * (reliability - 0.5)
        + 0.18 * (usefulness - 0.35)
        + 0.08 * stats.avg_reward
    )
    stats.reliability = _clamp(reliability, 0.0, 1.0)
    stats.usefulness = _clamp(usefulness, 0.0, 1.0)
    stats.retrieval_boost = _clamp(boost, -0.20, 0.20)
