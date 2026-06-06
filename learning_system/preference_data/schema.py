from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PreferenceSource(StrEnum):
    EVAL_HARNESS = "eval_harness"
    LLM_JUDGE = "llm_judge"
    USER_FEEDBACK = "user_feedback"
    SYNTHETIC_PAIR = "synthetic_pair"
    PROACTIVE_JUDGE = "proactive_judge"


class TaskType(StrEnum):
    RAG_QA = "rag_qa"
    TOOL_USE = "tool_use"
    MEMORY_QA = "memory_qa"
    INTERVIEW_QA = "interview_qa"
    PROACTIVE_REPLY = "proactive_reply"
    GENERAL_CHAT = "general_chat"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PreferenceSample(BaseModel):
    """A traceable pairwise preference sample for DPO-style post-training."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    source: PreferenceSource
    task_type: TaskType = TaskType.GENERAL_CHAT
    judge_score_chosen: float = Field(ge=0.0, le=1.0)
    judge_score_rejected: float = Field(ge=0.0, le=1.0)
    score_gap: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("sample_id", "prompt", "chosen", "rejected", "created_at")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @model_validator(mode="after")
    def _validate_pair(self) -> "PreferenceSample":
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected must differ")
        expected_gap = round(
            max(0.0, float(self.judge_score_chosen) - float(self.judge_score_rejected)),
            6,
        )
        supplied_gap = round(float(self.score_gap), 6)
        if abs(supplied_gap - expected_gap) > 1e-4:
            object.__setattr__(self, "score_gap", expected_gap)
        return self

    def to_trl_record(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class UserFeedbackEvent(BaseModel):
    """Minimal explicit user feedback schema for offline preference mining."""

    model_config = ConfigDict(extra="allow")

    feedback_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    feedback: str = Field(min_length=1)
    conversation_id: str = ""
    turn_id: str = ""
    candidate_id: str = ""
    selected_candidate_id: str = ""
    rejected_candidate_id: str = ""
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("feedback_id", "prompt", "response", "feedback")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("feedback")
    @classmethod
    def _normalize_feedback(cls, value: str) -> str:
        text = value.strip().lower()
        aliases = {
            "up": "thumbs_up",
            "+1": "thumbs_up",
            "like": "thumbs_up",
            "down": "thumbs_down",
            "-1": "thumbs_down",
            "dislike": "thumbs_down",
            "a": "choose_a",
            "b": "choose_b",
        }
        return aliases.get(text, text)

    @property
    def preference_score(self) -> float:
        if self.score is not None:
            return float(self.score)
        if self.feedback in {"thumbs_up", "choose_a", "choose_b", "selected"}:
            return 1.0
        if self.feedback in {"thumbs_down", "rejected"}:
            return 0.0
        return 0.5


def filter_samples(
    samples: list[PreferenceSample],
    *,
    min_score_gap: float = 0.2,
) -> list[PreferenceSample]:
    threshold = max(0.0, float(min_score_gap))
    return [sample for sample in samples if sample.score_gap >= threshold]

