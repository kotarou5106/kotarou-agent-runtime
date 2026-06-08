from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolRewardSignal:
    success: bool
    latency_ms: int = 0
    error_type: str = ""
    user_feedback: str = ""
    llm_judge_score: float | None = None
    task_completed: bool = False
    reward: float | None = None
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_reward(self) -> float:
        if self.reward is not None:
            return float(self.reward)

        reward = 1.0 if self.success else -1.0
        error = self.error_type.strip().lower()
        feedback = self.user_feedback.strip().lower()

        if error in {"tool_loop_guard", "loop_guard"}:
            reward = -0.5
        elif error in {"timeout", "safety_denial", "safety", "blocked"}:
            reward -= 0.5

        if feedback in {"thumbs_up", "up", "+1", "like"}:
            reward += 2.0
        elif feedback in {"thumbs_down", "down", "-1", "dislike"}:
            reward -= 2.0

        if self.llm_judge_score is not None:
            score = max(0.0, min(1.0, float(self.llm_judge_score)))
            reward += (score - 0.5) * 2.0

        if self.task_completed:
            reward += 1.0

        if self.latency_ms > 30_000:
            reward -= 0.25

        return round(reward, 6)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reward"] = self.compute_reward()
        return data

    @classmethod
    def from_tool_status(
        cls,
        *,
        status: str,
        latency_ms: int = 0,
        error_type: str = "",
        user_feedback: str = "",
        llm_judge_score: float | None = None,
        task_completed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolRewardSignal":
        normalized_status = str(status or "").strip().lower()
        success = normalized_status == "success"
        inferred_error = error_type
        if not inferred_error and normalized_status not in {"", "success"}:
            inferred_error = normalized_status
        return cls(
            success=success,
            latency_ms=max(0, int(latency_ms)),
            error_type=inferred_error,
            user_feedback=user_feedback,
            llm_judge_score=llm_judge_score,
            task_completed=task_completed,
            metadata=dict(metadata or {}),
        )
