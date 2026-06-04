from __future__ import annotations

import pytest

from evaluation_system.rag import RagEvalCase, evaluate_rag_case
from knowledge_system.injection import validate_answer_citations
from knowledge_system.indexing.store import KnowledgeStore


def test_rag_eval_recall_and_citation_hit() -> None:
    result = evaluate_rag_case(
        RagEvalCase(query="agent docs", gold_chunk_ids=["c1"], gold_document_ids=["d2"], k=2),
        retrieved_chunks=[
            {"chunk_id": "c1", "document_id": "d1"},
            {"chunk_id": "c9", "document_id": "d2"},
        ],
        answer="答案来自文档 [K1]",
        citations=[{"id": "K1", "chunk_id": "c1"}],
    )

    assert result.recall_at_k == 1.0
    assert result.citation_hit is True
    assert result.hit_chunk_ids == ["c1"]
    assert result.hit_document_ids == ["d2"]


def test_citation_validator_warns_on_unknown_id() -> None:
    result = validate_answer_citations(
        "这里引用了 [K1] 和 [K9]",
        [{"id": "K1", "chunk_id": "c1"}],
    )

    assert result.ok is False
    assert result.valid_ids == ["K1"]
    assert result.missing_ids == ["K9"]
    assert result.warnings


def test_knowledge_store_reports_vector_backend_fallback(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        assert store.vector_backend in {"sqlite-vec", "json-cosine"}
    finally:
        store.close()
