from __future__ import annotations

import math

from knowledge_system.evaluation.dataset import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalPrediction,
)

DEFAULT_K_VALUES = (1, 3, 5, 10)


def evaluate_retrieval_predictions(
    cases: list[RetrievalEvalCase],
    predictions: list[RetrievalEvalPrediction],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[dict[str, float], list[RetrievalEvalCaseResult]]:
    """Compute aggregate and per-case retrieval metrics."""
    prediction_by_case = {item.case_id: item for item in predictions}
    case_results = [
        evaluate_retrieval_case(case, prediction_by_case.get(case.case_id), k_values=k_values)
        for case in cases
    ]
    if not case_results:
        return _empty_metrics(k_values), []

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = _mean(result.recall_at_k.get(str(k), 0.0) for result in case_results)
        metrics[f"precision@{k}"] = _mean(result.precision_at_k.get(str(k), 0.0) for result in case_results)
        metrics[f"hit_rate@{k}"] = _mean(1.0 if result.hit_at_k.get(str(k), False) else 0.0 for result in case_results)
        metrics[f"ndcg@{k}"] = _mean(result.ndcg_at_k.get(str(k), 0.0) for result in case_results)
    metrics["mrr@5"] = _mean(result.reciprocal_rank for result in case_results)
    return metrics, case_results


def evaluate_retrieval_case(
    case: RetrievalEvalCase,
    prediction: RetrievalEvalPrediction | None,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> RetrievalEvalCaseResult:
    retrieved_chunks = list(prediction.retrieved_chunk_ids if prediction else [])
    retrieved_docs = list(prediction.retrieved_document_ids if prediction else [])
    relevant_chunks = {item for item in case.relevant_chunk_ids if item}
    relevant_docs = {item for item in case.relevant_document_ids if item}
    relevance = [
        _is_relevant(chunk_id, doc_id, relevant_chunks, relevant_docs)
        for chunk_id, doc_id in zip_longest(retrieved_chunks, retrieved_docs)
    ]

    hit_at_k: dict[str, bool] = {}
    recall_at_k: dict[str, float] = {}
    precision_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}
    total_relevant = max(1, len(relevant_chunks) + len(relevant_docs))
    for k in k_values:
        top_relevance = relevance[: max(1, int(k))]
        hits = sum(1 for item in top_relevance if item)
        hit_at_k[str(k)] = hits > 0
        recall_at_k[str(k)] = 0.0 if not (relevant_chunks or relevant_docs) else min(1.0, hits / total_relevant)
        precision_at_k[str(k)] = hits / max(1, int(k))
        ndcg_at_k[str(k)] = ndcg_at_k_score(relevance, k=k)

    first_rank = first_relevant_rank(relevance, k=5)
    return RetrievalEvalCaseResult(
        case_id=case.case_id,
        query=case.query,
        retrieved_chunk_ids=retrieved_chunks,
        retrieved_document_ids=retrieved_docs,
        relevant_chunk_ids=list(case.relevant_chunk_ids),
        relevant_document_ids=list(case.relevant_document_ids),
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        ndcg_at_k=ndcg_at_k,
        first_relevant_rank=first_rank,
        trace_id=prediction.trace_id if prediction else None,
        error=prediction.error if prediction else "missing prediction",
    )


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[: max(1, int(k))]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    top = retrieved_ids[: max(1, int(k))]
    if not top:
        return 0.0
    return len(set(top) & relevant_ids) / max(1, int(k))


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    return 1.0 if set(retrieved_ids[: max(1, int(k))]) & relevant_ids else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    for rank, item in enumerate(retrieved_ids[: max(1, int(k))], start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k_ids(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    relevance = [item in relevant_ids for item in retrieved_ids]
    return ndcg_at_k_score(relevance, k=k)


def ndcg_at_k_score(relevance: list[bool], *, k: int) -> float:
    top = relevance[: max(1, int(k))]
    dcg = sum((1.0 if rel else 0.0) / math.log2(rank + 1) for rank, rel in enumerate(top, start=1))
    ideal_hits = min(sum(1 for item in relevance if item), max(1, int(k)))
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def first_relevant_rank(relevance: list[bool], *, k: int) -> int | None:
    for rank, item in enumerate(relevance[: max(1, int(k))], start=1):
        if item:
            return rank
    return None


def zip_longest(left: list[str], right: list[str]) -> list[tuple[str, str]]:
    size = max(len(left), len(right))
    return [
        (
            left[index] if index < len(left) else "",
            right[index] if index < len(right) else "",
        )
        for index in range(size)
    ]


def _is_relevant(
    chunk_id: str,
    document_id: str,
    relevant_chunks: set[str],
    relevant_docs: set[str],
) -> bool:
    return bool((chunk_id and chunk_id in relevant_chunks) or (document_id and document_id in relevant_docs))


def _empty_metrics(k_values: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = 0.0
        metrics[f"precision@{k}"] = 0.0
        metrics[f"hit_rate@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
    metrics["mrr@5"] = 0.0
    return metrics


def _mean(values) -> float:
    items = [float(item) for item in values]
    return sum(items) / len(items) if items else 0.0
