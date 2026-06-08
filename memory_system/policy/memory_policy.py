from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from memory_system.policy.policy_store import MemoryPolicyStore
from memory_system.policy.reward_signal import MemoryPolicyEvent

ENABLE_ENV = "ENABLE_MEMORY_POLICY_OPTIMIZATION"
STORE_ENV = "MEMORY_POLICY_STORE_PATH"


class MemoryPolicy:
    """Applies feedback-derived retrieval boosts to memory search results."""

    def __init__(self, store: MemoryPolicyStore, *, max_abs_boost: float = 0.20) -> None:
        self.store = store
        self.max_abs_boost = max(0.0, min(0.5, float(max_abs_boost)))

    def rank_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for index, item in enumerate(results):
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                ranked.append(dict(item))
                continue
            stats = self.store.get_memory_stats(item_id)
            boost = max(-self.max_abs_boost, min(self.max_abs_boost, stats.retrieval_boost))
            base_score = _float_score(item.get("score"))
            updated = dict(item)
            updated["score"] = round(base_score + boost, 4)
            score_debug = dict(updated.get("_score_debug") or {})
            score_debug["before_memory_policy"] = round(base_score, 4)
            score_debug["memory_policy_boost"] = round(boost, 4)
            score_debug["memory_policy_reliability"] = round(stats.reliability, 4)
            score_debug["memory_policy_usefulness"] = round(stats.usefulness, 4)
            updated["_score_debug"] = score_debug
            updated["_memory_policy_rank_before"] = index + 1
            ranked.append(updated)
        ranked.sort(key=lambda item: _float_score(item.get("score")), reverse=True)
        for index, item in enumerate(ranked):
            item["_memory_policy_rank_after"] = index + 1
        return ranked

    def record_event(
        self,
        memory_id: str,
        event_type: str,
        *,
        reward: float | None = None,
        user_feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = MemoryPolicyEvent(
            memory_id=memory_id,
            event_type=event_type,
            reward=reward,
            user_feedback=user_feedback,
            metadata=dict(metadata or {}),
        )
        self.store.update(memory_id, event)

    def record_events(
        self,
        memory_ids: list[str],
        event_type: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        seen: set[str] = set()
        for memory_id in memory_ids:
            clean_id = str(memory_id or "").strip()
            if not clean_id or clean_id in seen:
                continue
            seen.add(clean_id)
            self.record_event(clean_id, event_type, metadata=metadata)


def memory_policy_from_env(default_path: str | Path | None = None) -> MemoryPolicy | None:
    enabled = os.getenv(ENABLE_ENV, "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    raw_path = os.getenv(STORE_ENV, "").strip()
    path = Path(raw_path) if raw_path else Path(default_path or ".memory_policy_stats.json")
    return MemoryPolicy(MemoryPolicyStore(path))


def _float_score(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect memory policy optimization stats.")
    parser.add_argument("--store", default=os.getenv(STORE_ENV, ".memory_policy_stats.json"))
    parser.add_argument("--stats", action="store_true", help="Print policy stats as JSON.")
    parser.add_argument("--reset", action="store_true", help="Reset policy stats.")
    args = parser.parse_args(argv)

    store = MemoryPolicyStore(args.store)
    if args.reset:
        store.reset()
        print(json.dumps({"reset": True, "store": str(store.path)}, ensure_ascii=False))
        return 0
    if args.stats or not args.reset:
        print(
            json.dumps(
                {"memories": [item.to_dict() for item in store.list_stats()]},
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
