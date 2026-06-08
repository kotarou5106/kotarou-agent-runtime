from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_EVENT_REWARDS = {
    "retrieved": 0.05,
    "injected": 0.10,
    "used": 0.50,
    "corrected": -1.00,
    "superseded": -0.75,
    "deleted": -1.00,
    "user_positive": 2.00,
    "user_negative": -2.00,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_memory_reward(
    event_type: str,
    *,
    user_feedback: str | None = None,
    explicit_reward: float | None = None,
) -> float:
    """Return a small, explainable reward for one memory feedback event."""
    if explicit_reward is not None:
        return float(explicit_reward)
    feedback = (user_feedback or "").strip().lower()
    if feedback in {"thumbs_up", "up", "positive", "like"}:
        return _EVENT_REWARDS["user_positive"]
    if feedback in {"thumbs_down", "down", "negative", "dislike"}:
        return _EVENT_REWARDS["user_negative"]
    return float(_EVENT_REWARDS.get(event_type, 0.0))


@dataclass(frozen=True)
class MemoryPolicyEvent:
    memory_id: str
    event_type: str
    reward: float | None = None
    user_feedback: str | None = None
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_reward(self) -> float:
        return calculate_memory_reward(
            self.event_type,
            user_feedback=self.user_feedback,
            explicit_reward=self.reward,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "event_type": self.event_type,
            "reward": self.resolved_reward(),
            "user_feedback": self.user_feedback,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
