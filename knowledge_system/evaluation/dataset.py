from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_document_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class RetrievalEvalDataset:
    name: str
    description: str = ""
    cases: list[RetrievalEvalCase] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalEvalPrediction:
    case_id: str
    query: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_document_ids: list[str] = field(default_factory=list)
    scores: list[float | None] = field(default_factory=list)
    trace_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RetrievalEvalCaseResult:
    case_id: str
    query: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_document_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_document_ids: list[str] = field(default_factory=list)
    hit_at_k: dict[str, bool] = field(default_factory=dict)
    recall_at_k: dict[str, float] = field(default_factory=dict)
    precision_at_k: dict[str, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[str, float] = field(default_factory=dict)
    first_relevant_rank: int | None = None
    trace_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RetrievalEvalResult:
    retriever_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    case_results: list[RetrievalEvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_dataset(path: str | Path) -> RetrievalEvalDataset:
    """Load a retrieval evaluation dataset from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [
        RetrievalEvalCase(
            case_id=str(item.get("case_id") or ""),
            query=str(item.get("query") or ""),
            relevant_chunk_ids=_strings(item.get("relevant_chunk_ids")),
            relevant_document_ids=_strings(item.get("relevant_document_ids")),
            expected_keywords=_strings(item.get("expected_keywords")),
            notes=(str(item.get("notes")) if item.get("notes") is not None else None),
        )
        for item in payload.get("cases", [])
    ]
    return RetrievalEvalDataset(
        name=str(payload.get("name") or Path(path).stem),
        description=str(payload.get("description") or ""),
        cases=cases,
    )


def save_eval_result(result: RetrievalEvalResult, path: str | Path) -> None:
    """Save one evaluation result as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]
