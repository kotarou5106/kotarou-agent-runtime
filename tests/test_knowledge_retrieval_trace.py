from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.chunking import DocumentChunker
from knowledge_system.indexing.indexer import KnowledgeIndexer
from knowledge_system.indexing.models import DocumentChunk, KnowledgeChunkHit, LoadedDocument
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.injection import (
    build_citations,
    build_knowledge_prompt_block,
    validate_answer_citations,
)
from knowledge_system.loading import DocumentLoader
from knowledge_system.retrieval import KnowledgeQuery
from knowledge_system.retrieval.bm25 import BM25Index
from knowledge_system.retrieval.retriever import KnowledgeRetriever, rrf_merge
from knowledge_system.retrieval.trace import (
    CitationValidationTrace,
    KnowledgeRetrievalTrace,
    KnowledgeRetrieverResult,
    KnowledgeTraceCollector,
    TraceHit,
)
from knowledge_system.retrieval.trace_formatter import (
    citation_blocks_to_trace,
    citation_validation_to_trace,
)
from knowledge_system.service import KnowledgeService


def test_trace_dataclasses_convert_to_json_with_preview_limit() -> None:
    collector = KnowledgeTraceCollector(trace_id="t1", preview_chars=80)
    trace = collector.start("工具调用")
    collector.record_vector_hits("工具调用", [_hit("c1", text="x" * 200)])
    collector.record_citation_validation(
        CitationValidationTrace(is_valid=True, message="ok")
    )

    payload = trace.to_dict()

    assert payload["trace_id"] == "t1"
    assert len(payload["vector_hits_by_query"]["工具调用"][0]["text_preview"]) <= 80
    assert json.loads(json.dumps(payload, ensure_ascii=False))["original_query"] == "工具调用"


def test_query_rewrite_trace_records_original_and_variants() -> None:
    collector = KnowledgeTraceCollector(trace_id="t2")
    trace = collector.start("工具怎么调用")
    collector.record_query_variants(["工具怎么调用", "工具怎么调用 tool calling"])

    assert trace.original_query == "工具怎么调用"
    assert trace.query_variants[0] == "工具怎么调用"


def test_raw_hits_trace_records_vector_and_bm25_by_query() -> None:
    collector = KnowledgeTraceCollector(trace_id="t3")
    trace = collector.start("memory")
    collector.record_vector_hits("memory", [_hit("c1", score=0.8, lanes=("vector",))])
    collector.record_bm25_hits("memory", [_hit("c2", score=2.1, lanes=("bm25",))])

    vector_hit = trace.vector_hits_by_query["memory"][0]
    bm25_hit = trace.bm25_hits_by_query["memory"][0]
    assert vector_hit.chunk_id == "c1"
    assert vector_hit.rank == 1
    assert vector_hit.score == 0.8
    assert vector_hit.source == "vector"
    assert bm25_hit.source == "bm25"


def test_rrf_trace_records_sources_and_rank() -> None:
    merged = rrf_merge(
        [
            ("vector", [_hit("shared", score=0.9, lanes=("vector",)), _hit("v2", score=0.7)]),
            ("bm25", [_hit("shared", score=4.0, lanes=("bm25",)), _hit("b2", score=3.0, lanes=("bm25",))]),
        ],
        top_k=3,
    )
    collector = KnowledgeTraceCollector(trace_id="t4")
    trace = collector.start("shared")
    collector.record_rrf_hits(merged)

    assert trace.rrf_merged_hits[0].chunk_id == "shared"
    assert trace.rrf_merged_hits[0].rank == 1
    assert trace.rrf_merged_hits[0].sources == ["bm25", "vector"]


def test_citation_trace_records_success_and_failure() -> None:
    citations = build_citations([_hit("c1", score=1.0)])
    ok_block, _ = build_knowledge_prompt_block([_hit("c1", score=1.0)], citations)
    ok = citation_validation_to_trace(
        validate_answer_citations(ok_block, citations),
        citations,
    )
    failed = CitationValidationTrace(
        is_valid=False,
        missing_citations=["K9"],
        invalid_citations=["K9"],
        message="citation validation failed",
    )

    collector = KnowledgeTraceCollector(trace_id="t5")
    trace = collector.start("citation")
    collector.record_citation_blocks(citation_blocks_to_trace(citations))
    collector.record_citation_validation(ok)
    assert trace.citation_validation is not None
    assert trace.citation_validation.is_valid is True

    collector.record_citation_validation(failed)
    assert trace.citation_validation is not None
    assert trace.citation_validation.is_valid is False
    assert trace.warnings


@pytest.mark.asyncio
async def test_retriever_include_trace_preserves_default_behavior(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_store(store, tmp_path)
        retriever = KnowledgeRetriever(
            store=store,
            embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
            top_k=3,
            score_threshold=0.01,
        )

        plain = await retriever.retrieve("工具调用")
        traced = await retriever.retrieve("工具调用", include_trace=True)
        empty = await retriever.retrieve("", include_trace=True)

        assert isinstance(plain, list)
        assert plain
        assert isinstance(traced, KnowledgeRetrieverResult)
        assert traced.hits
        assert traced.trace.original_query == "工具调用"
        assert "工具调用" in traced.trace.query_variants
        assert traced.trace.vector_hits_by_query
        assert traced.trace.bm25_hits_by_query
        assert traced.trace.rrf_merged_hits
        assert traced.trace.selected_chunks
        assert isinstance(empty, KnowledgeRetrieverResult)
        assert empty.hits == []
        assert "empty query" in empty.trace.warnings
    finally:
        store.close()


@pytest.mark.asyncio
async def test_service_persists_full_knowledge_trace_for_dashboard(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_store(store, tmp_path)
        service = KnowledgeService(
            store=store,
            indexer=KnowledgeIndexer(
                store=store,
                loader=DocumentLoader(),
                chunker=DocumentChunker(),
                embedder=None,
            ),
            retriever=KnowledgeRetriever(
                store=store,
                embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
                top_k=3,
                score_threshold=0.01,
            ),
        )

        result = await service.retrieve(KnowledgeQuery(text="工具怎么调用", top_k=3))
        events = store.list_retrieval_events(limit=1)

        assert result.hits
        assert result.trace["query_variants"]
        assert result.trace["citation_ids"]
        assert result.trace["citation_validation"]["is_valid"] is True
        assert result.trace["stage_timings_ms"]["prompt_injection"] >= 0
        assert events
        assert events[0]["trace"]["original_query"] == "工具怎么调用"
        assert events[0]["trace"]["bm25_hits_by_query"]
    finally:
        store.close()


def test_bm25_trace_hit_preview_works_with_index_rows() -> None:
    hits = BM25Index([_row("c1", "d1", "知识检索支持 BM25 sparse retrieval")]).search(
        "BM25"
    )
    collector = KnowledgeTraceCollector(trace_id="t6")
    trace = collector.start("BM25")
    collector.record_bm25_hits("BM25", hits)

    assert trace.bm25_hits_by_query["BM25"][0].chunk_id == "c1"


def _hit(
    chunk_id: str,
    *,
    score: float = 1.0,
    text: str = "Agent Runtime trace chunk",
    lanes: tuple[str, ...] = ("vector",),
) -> KnowledgeChunkHit:
    return KnowledgeChunkHit(
        chunk_id=chunk_id,
        document_id=f"d_{chunk_id}",
        text=text,
        score=score,
        source_path=f"/tmp/{chunk_id}.md",
        title=chunk_id,
        lanes=lanes,
        sources=lanes,
    )


def _row(chunk_id: str, document_id: str, text: str) -> dict[str, object]:
    return {
        "id": chunk_id,
        "document_id": document_id,
        "text": text,
        "source_path": f"/tmp/{document_id}.md",
        "title": document_id,
        "heading_path": "",
        "line_start": 1,
        "line_end": 1,
    }


def _seed_store(store: KnowledgeStore, tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            id="c_tool",
            document_id="d_tool",
            chunk_index=0,
            text="Tool calling and function calling are exposed through plugins and lifecycle hooks.",
            content_hash="h_tool",
        ),
        DocumentChunk(
            id="c_memory",
            document_id="d_memory",
            chunk_index=0,
            text="Long-term memory retrieval brings durable user facts into the prompt.",
            content_hash="h_memory",
        ),
    ]
    docs = [
        ("d_tool", "tool.md", [chunks[0]], {"c_tool": [1.0, 0.0]}),
        ("d_memory", "memory.md", [chunks[1]], {"c_memory": [0.0, 1.0]}),
    ]
    for document_id, filename, doc_chunks, vectors in docs:
        store.upsert_document_with_chunks(
            document_id=document_id,
            document=LoadedDocument(
                source_path=tmp_path / filename,
                title=filename,
                content="\n".join(chunk.text for chunk in doc_chunks),
                file_type="markdown",
                content_hash=f"h_{document_id}",
                mtime=0.0,
            ),
            chunks=doc_chunks,
            vectors=vectors,
            embedding_model="trace-test",
        )


class _KeywordEmbedder:
    async def embed(self, text: str) -> list[float]:
        raw = text.lower()
        return [
            1.0 if "tool" in raw or "function" in raw or "工具" in raw else 0.0,
            1.0 if "memory" in raw or "记忆" in raw else 0.0,
        ]
