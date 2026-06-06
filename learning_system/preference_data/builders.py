from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from learning_system.preference_data.schema import (
    PreferenceSample,
    PreferenceSource,
    TaskType,
    UserFeedbackEvent,
    filter_samples,
)


def stable_sample_id(*parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"pref_{digest}"


def infer_task_type(record: dict[str, Any]) -> TaskType:
    raw = str(
        record.get("task_type")
        or record.get("question_type")
        or record.get("scenario_name")
        or record.get("scenario_id")
        or ""
    ).lower()
    if "rag" in raw or "retrieval" in raw or "knowledge" in raw:
        return TaskType.RAG_QA
    if "memory" in raw or "persona" in raw or "longmem" in raw:
        return TaskType.MEMORY_QA
    if "tool" in raw or "shell" in raw or "spawn" in raw:
        return TaskType.TOOL_USE
    if "interview" in raw:
        return TaskType.INTERVIEW_QA
    if "proactive" in raw:
        return TaskType.PROACTIVE_REPLY
    return TaskType.GENERAL_CHAT


def normalize_score(record: dict[str, Any]) -> float:
    for key in (
        "judge_score",
        "score",
        "accuracy",
        "judge_acc",
        "final_score",
        "preference_score",
    ):
        raw = record.get(key)
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, int | float):
            return max(0.0, min(1.0, float(raw)))
    if record.get("judge_correct") is True or record.get("is_correct") is True:
        return 1.0
    if record.get("judge_correct") is False or record.get("is_correct") is False:
        return 0.0
    if record.get("passed") is True:
        return 1.0
    if record.get("passed") is False:
        return 0.0
    return 0.5


def extract_prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "user_prompt", "user_message"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("input_messages")
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content") or "").strip()
            for item in messages
            if isinstance(item, dict) and item.get("content")
        ).strip()
    return ""


def extract_response(record: dict[str, Any]) -> str:
    for key in (
        "response",
        "answer",
        "final_answer",
        "predicted_answer",
        "candidate",
        "completion",
        "text",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    summary = record.get("final_answer_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return ""


def _metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {}
    for key in (
        "tools_used",
        "tool_chain",
        "retrieved_docs",
        "retrieved_chunk_ids",
        "citations",
        "conversation_id",
        "turn_id",
        "scenario_name",
        "scenario_id",
        "trace_id",
        "backend",
    ):
        if key in record and key not in metadata:
            metadata[key] = record[key]
    return metadata


@dataclass(frozen=True)
class CandidateRecord:
    prompt: str
    response: str
    score: float
    source_record: dict[str, Any]


class EvaluationHarnessPreferenceBuilder:
    """Build pairwise preferences from eval runs grouped by prompt/scenario."""

    def build(
        self,
        records: Iterable[dict[str, Any]],
        *,
        min_score_gap: float = 0.2,
    ) -> list[PreferenceSample]:
        grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
        for record in records:
            prompt = extract_prompt(record)
            response = extract_response(record)
            if not prompt or not response:
                continue
            group_key = str(
                record.get("prompt_id")
                or record.get("case_id")
                or record.get("scenario_name")
                or record.get("scenario_id")
                or prompt
            )
            grouped[group_key].append(
                CandidateRecord(
                    prompt=prompt,
                    response=response,
                    score=normalize_score(record),
                    source_record=record,
                )
            )

        samples: list[PreferenceSample] = []
        for group_key, candidates in grouped.items():
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
            chosen = ordered[0]
            rejected = ordered[-1]
            if chosen.response.strip() == rejected.response.strip():
                continue
            sample = PreferenceSample(
                sample_id=stable_sample_id("eval", group_key, chosen.response, rejected.response),
                prompt=chosen.prompt,
                chosen=chosen.response,
                rejected=rejected.response,
                source=PreferenceSource.EVAL_HARNESS,
                task_type=infer_task_type(chosen.source_record),
                judge_score_chosen=chosen.score,
                judge_score_rejected=rejected.score,
                score_gap=max(0.0, chosen.score - rejected.score),
                metadata={
                    "group_key": group_key,
                    "chosen": _metadata_from_record(chosen.source_record),
                    "rejected": _metadata_from_record(rejected.source_record),
                },
            )
            samples.append(sample)
        return filter_samples(samples, min_score_gap=min_score_gap)


class LLMJudgePreferenceBuilder:
    """Build preferences from pairwise judge records or scalar scored candidates."""

    def build(
        self,
        records: Iterable[dict[str, Any]],
        *,
        min_score_gap: float = 0.2,
    ) -> list[PreferenceSample]:
        samples: list[PreferenceSample] = []
        for record in records:
            prompt = extract_prompt(record)
            if not prompt:
                continue
            response_a = str(record.get("response_a") or record.get("candidate_a") or "").strip()
            response_b = str(record.get("response_b") or record.get("candidate_b") or "").strip()
            if not response_a or not response_b or response_a == response_b:
                continue
            score_a = _coerce_score(record.get("score_a"), default=None)
            score_b = _coerce_score(record.get("score_b"), default=None)
            if score_a is None or score_b is None:
                winner = str(record.get("winner") or record.get("chosen") or "").strip().lower()
                score_a, score_b = _scores_from_winner(winner)
            if score_a >= score_b:
                chosen, rejected = response_a, response_b
                chosen_score, rejected_score = score_a, score_b
            else:
                chosen, rejected = response_b, response_a
                chosen_score, rejected_score = score_b, score_a
            samples.append(
                PreferenceSample(
                    sample_id=stable_sample_id("judge", prompt, chosen, rejected),
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    source=PreferenceSource.LLM_JUDGE,
                    task_type=infer_task_type(record),
                    judge_score_chosen=chosen_score,
                    judge_score_rejected=rejected_score,
                    score_gap=max(0.0, chosen_score - rejected_score),
                    metadata=_metadata_from_record(record),
                )
            )
        return filter_samples(samples, min_score_gap=min_score_gap)


class UserFeedbackPreferenceBuilder:
    """Convert explicit thumbs-up/down or A/B choices into DPO pairs."""

    def build(
        self,
        records: Iterable[dict[str, Any]],
        *,
        min_score_gap: float = 0.2,
    ) -> list[PreferenceSample]:
        events = [UserFeedbackEvent.model_validate(record) for record in records]
        grouped: dict[str, list[UserFeedbackEvent]] = defaultdict(list)
        for event in events:
            key = event.conversation_id or event.turn_id or event.prompt
            grouped[key].append(event)

        samples: list[PreferenceSample] = []
        for key, group in grouped.items():
            positives = [item for item in group if item.preference_score > 0.5]
            negatives = [item for item in group if item.preference_score < 0.5]
            if not positives or not negatives:
                continue
            chosen = max(positives, key=lambda item: item.preference_score)
            rejected = min(negatives, key=lambda item: item.preference_score)
            if chosen.response.strip() == rejected.response.strip():
                continue
            samples.append(
                PreferenceSample(
                    sample_id=stable_sample_id("feedback", key, chosen.response, rejected.response),
                    prompt=chosen.prompt,
                    chosen=chosen.response,
                    rejected=rejected.response,
                    source=PreferenceSource.USER_FEEDBACK,
                    task_type=TaskType.GENERAL_CHAT,
                    judge_score_chosen=chosen.preference_score,
                    judge_score_rejected=rejected.preference_score,
                    score_gap=max(0.0, chosen.preference_score - rejected.preference_score),
                    metadata={
                        "group_key": key,
                        "conversation_id": chosen.conversation_id or rejected.conversation_id,
                        "turn_id": chosen.turn_id or rejected.turn_id,
                        "chosen_feedback_id": chosen.feedback_id,
                        "rejected_feedback_id": rejected.feedback_id,
                        "chosen_metadata": chosen.metadata,
                        "rejected_metadata": rejected.metadata,
                    },
                )
            )
        return filter_samples(samples, min_score_gap=min_score_gap)


def _coerce_score(raw: object, *, default: float | None) -> float | None:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, int | float):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return default
    return default


def _scores_from_winner(winner: str) -> tuple[float, float]:
    if winner in {"a", "response_a", "candidate_a", "chosen_a"}:
        return 1.0, 0.0
    if winner in {"b", "response_b", "candidate_b", "chosen_b"}:
        return 0.0, 1.0
    return 0.5, 0.5


def score_text_against_reference(response: str, reference: str) -> float:
    """Small deterministic judge for dry-run and tests, not a reward model."""
    response_tokens = _tokens(response)
    reference_tokens = _tokens(reference)
    if not response_tokens or not reference_tokens:
        return 0.0
    overlap = len(response_tokens & reference_tokens)
    precision = overlap / max(1, len(response_tokens))
    recall = overlap / max(1, len(reference_tokens))
    if precision + recall == 0:
        return 0.0
    return max(0.0, min(1.0, 2 * precision * recall / (precision + recall)))


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"\W+", text.lower()) if token}


def created_at_from_record(record: dict[str, Any]) -> str:
    raw = record.get("created_at") or record.get("timestamp")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return datetime.now(timezone.utc).isoformat()

