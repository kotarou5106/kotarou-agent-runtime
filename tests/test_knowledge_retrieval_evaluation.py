from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.evaluation.dataset import (
    RetrievalEvalCase,
    RetrievalEvalDataset,
    RetrievalEvalPrediction,
    load_eval_dataset,
    save_eval_result,
)
from knowledge_system.evaluation.metrics import (
    evaluate_retrieval_predictions,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k_ids,
    precision_at_k,
    recall_at_k,
)
from knowledge_system.evaluation.report import (
    render_markdown_report,
    write_json_report,
    write_markdown_report,
)
from knowledge_system.evaluation.runner import run_retrieval_evaluation
from knowledge_system.indexing.models import DocumentChunk, LoadedDocument
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.retrieval import KnowledgeRetriever


def test_eval_dataset_loads_from_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "name": "sample",
                "description": "desc",
                "cases": [
                    {
                        "case_id": "c1",
                        "query": "tool calling",
                        "relevant_chunk_ids": ["chunk_tool"],
                        "relevant_document_ids": ["doc_tool"],
                        "expected_keywords": ["tool"],
                        "notes": "note",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_eval_dataset(path)

    assert dataset.name == "sample"
    assert dataset.cases[0].case_id == "c1"
    assert dataset.cases[0].relevant_chunk_ids == ["chunk_tool"]


def test_eval_dataset_handles_empty_cases(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text('{"name": "empty", "cases": []}', encoding="utf-8")

    dataset = load_eval_dataset(path)
    metrics, case_results = evaluate_retrieval_predictions(dataset.cases, [])

    assert dataset.cases == []
    assert case_results == []
    assert metrics["recall@5"] == 0.0


def test_retrieval_metrics_are_correct() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0
    assert mrr_at_k(retrieved, relevant, k=5) == 0.5
    assert ndcg_at_k_ids(retrieved, relevant, k=4) > 0.0
    assert recall_at_k(retrieved, set(), k=3) == 0.0


def test_aggregate_metrics_include_expected_keys() -> None:
    dataset = RetrievalEvalDataset(
        name="unit",
        cases=[
            RetrievalEvalCase(case_id="case1", query="q", relevant_chunk_ids=["c2"]),
            RetrievalEvalCase(case_id="case2", query="q", relevant_chunk_ids=["missing"]),
        ],
    )
    predictions = [
        RetrievalEvalPrediction(
            case_id="case1",
            query="q",
            retrieved_chunk_ids=["c1", "c2"],
            retrieved_document_ids=["d1", "d2"],
        ),
        RetrievalEvalPrediction(
            case_id="case2",
            query="q",
            retrieved_chunk_ids=["c9"],
            retrieved_document_ids=["d9"],
        ),
    ]

    metrics, case_results = evaluate_retrieval_predictions(dataset.cases, predictions)

    assert metrics["hit_rate@3"] == 0.5
    assert metrics["mrr@5"] == 0.25
    assert case_results[0].first_relevant_rank == 2


@pytest.mark.asyncio
async def test_runner_evaluates_bm25_vector_and_hybrid(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_eval_store(store, tmp_path)
        retriever = KnowledgeRetriever(
            store=store,
            embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
            top_k=5,
            score_threshold=0.01,
        )
        dataset = _eval_dataset()

        bm25 = await run_retrieval_evaluation(dataset, retriever, mode="bm25", top_k=5)
        vector = await run_retrieval_evaluation(dataset, retriever, mode="vector", top_k=5)
        hybrid = await run_retrieval_evaluation(dataset, retriever, mode="hybrid_rrf", top_k=5)

        assert bm25.retriever_name == "bm25"
        assert vector.retriever_name == "vector"
        assert hybrid.retriever_name == "hybrid_rrf"
        assert bm25.metrics["hit_rate@5"] > 0.0
        assert vector.metrics["hit_rate@5"] > 0.0
        assert hybrid.metrics["hit_rate@5"] >= bm25.metrics["hit_rate@5"]
        assert len(hybrid.case_results) == len(dataset.cases)
        assert hybrid.case_results[-1].hit_at_k["5"] is False
        assert hybrid.case_results[0].trace_id
    finally:
        store.close()


def test_report_writes_json_and_markdown(tmp_path: Path) -> None:
    result = _dummy_result()
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    save_eval_result(result, tmp_path / "single.json")
    write_json_report([result], json_path)
    write_markdown_report([result], md_path)
    markdown = render_markdown_report([result])

    assert json.loads(json_path.read_text(encoding="utf-8"))["results"][0]["retriever_name"] == "bm25"
    assert "# Knowledge Retrieval Evaluation Report" in markdown
    assert "| Retriever | Recall@1 | Recall@3 | Recall@5 | Precision@5 | HitRate@5 | MRR@5 | NDCG@5 |" in md_path.read_text(encoding="utf-8")


def _eval_dataset() -> RetrievalEvalDataset:
    return RetrievalEvalDataset(
        name="runtime-mini",
        cases=[
            RetrievalEvalCase(
                case_id="tool",
                query="工具调用 function calling",
                relevant_chunk_ids=["chunk_tool"],
                relevant_document_ids=["doc_tool"],
            ),
            RetrievalEvalCase(
                case_id="memory",
                query="long term memory retrieval",
                relevant_chunk_ids=["chunk_memory"],
                relevant_document_ids=["doc_memory"],
            ),
            RetrievalEvalCase(
                case_id="scheduler",
                query="scheduler background task",
                relevant_chunk_ids=["chunk_scheduler"],
                relevant_document_ids=["doc_scheduler"],
            ),
            RetrievalEvalCase(
                case_id="miss",
                query="unrelated quantum banana",
                relevant_chunk_ids=["chunk_missing"],
                relevant_document_ids=["doc_missing"],
            ),
        ],
    )


def _dummy_result():
    dataset = RetrievalEvalDataset(
        name="dummy",
        cases=[RetrievalEvalCase(case_id="c1", query="tool", relevant_chunk_ids=["c1"])],
    )
    metrics, case_results = evaluate_retrieval_predictions(
        dataset.cases,
        [
            RetrievalEvalPrediction(
                case_id="c1",
                query="tool",
                retrieved_chunk_ids=["c1"],
                retrieved_document_ids=["d1"],
                trace_id="t1",
            )
        ],
    )
    from knowledge_system.evaluation.dataset import RetrievalEvalResult

    return RetrievalEvalResult(retriever_name="bm25", metrics=metrics, case_results=case_results)


def _seed_eval_store(store: KnowledgeStore, tmp_path: Path) -> None:
    docs = [
        (
            "doc_tool",
            "tool.md",
            "chunk_tool",
            "Tool calling and function calling execute registered tools from plugins.",
            [1.0, 0.0, 0.0],
        ),
        (
            "doc_memory",
            "memory.md",
            "chunk_memory",
            "Long-term memory retrieval brings durable user facts into context.",
            [0.0, 1.0, 0.0],
        ),
        (
            "doc_scheduler",
            "scheduler.md",
            "chunk_scheduler",
            "Scheduler background task workflows run proactive notifications.",
            [0.0, 0.0, 1.0],
        ),
    ]
    for document_id, filename, chunk_id, text, vector in docs:
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_index=0,
            text=text,
            content_hash=f"h_{chunk_id}",
        )
        store.upsert_document_with_chunks(
            document_id=document_id,
            document=LoadedDocument(
                source_path=tmp_path / filename,
                title=filename,
                content=text,
                file_type="markdown",
                content_hash=f"h_{document_id}",
                mtime=0.0,
            ),
            chunks=[chunk],
            vectors={chunk_id: vector},
            embedding_model="eval-test",
        )


class _KeywordEmbedder:
    async def embed(self, text: str) -> list[float]:
        raw = text.lower()
        return [
            1.0 if "tool" in raw or "function" in raw or "工具" in raw else 0.0,
            1.0 if "memory" in raw or "记忆" in raw else 0.0,
            1.0 if "scheduler" in raw or "task" in raw else 0.0,
        ]
