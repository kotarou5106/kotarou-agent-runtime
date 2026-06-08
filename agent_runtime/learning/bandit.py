from __future__ import annotations

import math
import random
from dataclasses import dataclass

from agent_runtime.learning.policy_store import ToolPolicyStore, ToolStats


@dataclass(frozen=True)
class ScoredTool:
    tool_name: str
    score: float
    stats: ToolStats
    exploration_bonus: float


class BanditPolicy:
    """UCB-style bandit for ranking candidate tools.

    Cold-start tools receive an explicit exploration score, so new tools remain
    discoverable instead of being permanently buried by historical winners.
    """

    def __init__(
        self,
        store: ToolPolicyStore,
        *,
        exploration_weight: float = 1.0,
        cold_start_score: float = 0.25,
        rng: random.Random | None = None,
    ) -> None:
        self._store = store
        self._exploration_weight = max(0.0, float(exploration_weight))
        self._cold_start_score = float(cold_start_score)
        self._rng = rng

    def score_tool(self, tool_name: str, *, total_count: int | None = None) -> ScoredTool:
        stats = self._store.get_tool_stats(tool_name)
        total = total_count
        if total is None:
            total = sum(item.count for item in self._store.list_stats())
        if stats.count <= 0:
            jitter = (self._rng.random() * 1e-6) if self._rng is not None else 0.0
            return ScoredTool(
                tool_name=tool_name,
                score=self._cold_start_score + self._exploration_weight + jitter,
                stats=stats,
                exploration_bonus=self._exploration_weight,
            )
        bonus = self._exploration_weight * math.sqrt(
            math.log(max(2, total + 1)) / stats.count
        )
        return ScoredTool(
            tool_name=tool_name,
            score=stats.avg_reward + bonus,
            stats=stats,
            exploration_bonus=bonus,
        )

    def rank(self, candidate_tools: list[str]) -> list[ScoredTool]:
        names = [name for name in dict.fromkeys(candidate_tools) if name]
        total = sum(item.count for item in self._store.list_stats())
        scored = [self.score_tool(name, total_count=total) for name in names]
        scored.sort(key=lambda item: (-item.score, item.tool_name))
        return scored
