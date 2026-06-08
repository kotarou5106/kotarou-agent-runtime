from __future__ import annotations

from typing import Literal

from knowledge_system.evaluation.dataset import (
    RetrievalEvalDataset,
    RetrievalEvalPrediction,
    RetrievalEvalResult,
)
from knowledge_system.evaluation.metrics import evaluate_retrieval_predictions
from knowledge_system.retrieval.trace import KnowledgeRetrieverResult
from knowledge_system.retrieval.retriever import KnowledgeRetriever

RetrievalMode = Literal["bm25", "vector", "hybrid_rrf"]


async def run_retrieval_evaluation(
    dataset: RetrievalEvalDataset,
    retriever: KnowledgeRetriever,
    *,
    mode: RetrievalMode = "hybrid_rrf",
    top_k: int = 5,
    include_trace: bool = True,
) -> RetrievalEvalResult:
    """Run retrieval evaluation for one retriever mode."""
    predictions: list[RetrievalEvalPrediction] = []
    for case in dataset.cases:
        try:
            retrieval = await retriever.retrieve(
                case.query,
                top_k=top_k,
                include_trace=include_trace,
                retrieval_mode=mode,
            )
            if isinstance(retrieval, KnowledgeRetrieverResult):
                hits = retrieval.hits
                trace_id = retrieval.trace.trace_id
            else:
                hits = retrieval
                trace_id = None
            predictions.append(
                RetrievalEvalPrediction(
                    case_id=case.case_id,
                    query=case.query,
                    retrieved_chunk_ids=[hit.chunk_id for hit in hits],
                    retrieved_document_ids=[hit.document_id for hit in hits],
                    scores=[hit.score for hit in hits],
                    trace_id=trace_id,
                )
            )
        except Exception as exc:
            predictions.append(
                RetrievalEvalPrediction(
                    case_id=case.case_id,
                    query=case.query,
                    error=str(exc),
                )
            )

    metrics, case_results = evaluate_retrieval_predictions(dataset.cases, predictions)
    return RetrievalEvalResult(
        retriever_name=mode,
        metrics=metrics,
        case_results=case_results,
    )
