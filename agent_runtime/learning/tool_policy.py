from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_runtime.learning.bandit import BanditPolicy
from agent_runtime.learning.policy_store import ToolPolicyStore
from agent_runtime.learning.reward_signal import ToolRewardSignal

ENABLE_ENV = "ENABLE_TOOL_POLICY_OPTIMIZATION"
STORE_ENV = "TOOL_POLICY_STORE_PATH"


@dataclass(frozen=True)
class ToolCallContext:
    user_intent: str = ""
    conversation_id: str = ""
    session_id: str = ""
    task_type: str = ""
    candidate_tools: list[str] = field(default_factory=list)
    retrieved_memory_count: int = 0
    retrieved_knowledge_count: int = 0
    previous_tool_failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolAction:
    tool_name: str
    source: str = ""
    rank_before_policy: int = 0
    rank_after_policy: int = 0


class ToolSelectionPolicy:
    def __init__(self, store: ToolPolicyStore, bandit: BanditPolicy | None = None) -> None:
        self.store = store
        self.bandit = bandit or BanditPolicy(store)

    def rank_tools(
        self,
        candidate_tools: list[str],
        *,
        context: ToolCallContext | None = None,
    ) -> list[str]:
        _ = context
        ranked = self.bandit.rank(candidate_tools)
        return [item.tool_name for item in ranked]

    def rerank_search_results(
        self,
        results: list[dict[str, Any]],
        *,
        context: ToolCallContext | None = None,
    ) -> list[dict[str, Any]]:
        if len(results) <= 1:
            return list(results)
        by_name = {
            str(item.get("name") or ""): dict(item)
            for item in results
            if str(item.get("name") or "")
        }
        ranked_names = self.rank_tools(list(by_name.keys()), context=context)
        ranked: list[dict[str, Any]] = []
        for after_rank, name in enumerate(ranked_names, start=1):
            item = by_name[name]
            before_rank = next(
                (index for index, old in enumerate(results, start=1) if old.get("name") == name),
                after_rank,
            )
            item["rank_before_policy"] = before_rank
            item["rank_after_policy"] = after_rank
            item["policy_score"] = self.bandit.score_tool(name).score
            ranked.append(item)
        return ranked

    def update(
        self,
        tool_name: str,
        signal: ToolRewardSignal,
    ):
        return self.store.update(
            tool_name,
            signal.compute_reward(),
            success=signal.success,
            created_at=signal.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.store.to_dict()


def is_tool_policy_enabled() -> bool:
    value = os.getenv(ENABLE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def default_policy_store_path() -> Path:
    raw = os.getenv(STORE_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".kotarou" / "workspace" / "tool_policy.json"


def build_default_tool_selection_policy(
    *,
    enabled: bool | None = None,
    store_path: str | Path | None = None,
) -> ToolSelectionPolicy | None:
    active = is_tool_policy_enabled() if enabled is None else bool(enabled)
    if not active:
        return None
    store = ToolPolicyStore(store_path or default_policy_store_path())
    return ToolSelectionPolicy(store)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or reset feedback-driven tool selection policy stats."
    )
    parser.add_argument("--stats", action="store_true", help="Print policy statistics as JSON.")
    parser.add_argument("--reset", action="store_true", help="Reset policy statistics.")
    parser.add_argument("--store", type=Path, default=None, help="Override policy store path.")
    args = parser.parse_args(argv)

    store = ToolPolicyStore(args.store or default_policy_store_path())
    if args.reset:
        store.reset()
        print(json.dumps({"reset": True, "store": str(store.path)}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(store.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
