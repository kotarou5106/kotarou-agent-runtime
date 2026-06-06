from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.models import DocumentChunk, KnowledgeChunkHit, LoadedDocument
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.injection import build_citations, build_knowledge_prompt_block
from knowledge_system.retrieval.bm25 import BM25Index
from knowledge_system.retrieval.query_rewrite import rewrite_query
from knowledge_system.retrieval.retriever import KnowledgeRetriever, rrf_merge


def test_bm25_exact_keyword_match_ranks_first() -> None:
    index = BM25Index(
        [
            _row("c1", "d1", "Agent Runtime supports tool calling hooks."),
            _row("c2", "d2", "Dashboard observability shows session events."),
        ]
    )

    hits = index.search("tool calling", top_k=2)

    assert [hit.chunk_id for hit in hits] == ["c1"]
    assert hits[0].score > 0


def test_bm25_rare_term_has_higher_score() -> None:
    index = BM25Index(
        [
            _row("c1", "d1", "agent agent agent runtime memory"),
            _row("c2", "d2", "agent runtime sqlite vec vector backend"),
        ]
    )

    hits = index.search("agent sqlite vec", top_k=2)

    assert hits[0].chunk_id == "c2"


def test_bm25_empty_query_returns_empty() -> None:
    assert BM25Index([_row("c1", "d1", "tool calling")]).search("") == []


def test_bm25_is_case_insensitive_for_english() -> None:
    hits = BM25Index([_row("c1", "d1", "Function Calling Tools")]).search(
        "function calling"
    )

    assert hits[0].chunk_id == "c1"


def test_bm25_supports_basic_chinese_keywords() -> None:
    hits = BM25Index([_row("c1", "d1", "知识检索支持工具调用和引用校验")]).search(
        "工具调用"
    )

    assert hits[0].chunk_id == "c1"


def test_query_rewrite_expands_tool_terms() -> None:
    variants = rewrite_query("工具调用怎么接入")

    assert variants[0] == "工具调用怎么接入"
    assert any("tool calling" in item and "function calling" in item for item in variants)


def test_query_rewrite_expands_memory_terms() -> None:
    variants = rewrite_query("长期记忆如何检索")

    assert variants[0] == "长期记忆如何检索"
    assert any("memory" in item and "long-term memory" in item for item in variants)


def test_query_rewrite_deduplicates_variants() -> None:
    variants = rewrite_query("memory")

    assert len(variants) == len(set(item.lower() for item in variants))


def test_rrf_boosts_chunks_found_by_vector_and_bm25() -> None:
    vector_hits = [_hit("c1", score=0.99), _hit("c2", score=0.80)]
    bm25_hits = [_hit("c2", score=4.0, lanes=("bm25",)), _hit("c3", score=3.0, lanes=("bm25",))]

    merged = rrf_merge([("vector", vector_hits), ("bm25", bm25_hits)], top_k=3)

    assert [hit.chunk_id for hit in merged] == ["c2", "c1", "c3"]
    assert len({hit.chunk_id for hit in merged}) == len(merged)
    assert merged[0].sources == ("bm25", "vector")
    assert merged[0].vector_rank == 2
    assert merged[0].bm25_rank == 1
    assert merged[0].rrf_score is not None


@pytest.mark.asyncio
async def test_hybrid_retriever_supports_vector_bm25_rewrite_and_citations(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_knowledge_store(store, tmp_path)
        retriever = KnowledgeRetriever(
            store=store,
            embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
            top_k=3,
            score_threshold=0.01,
        )

        semantic_hits = await retriever.retrieve("How does the agent runtime coordinate turns?")
        assert semantic_hits
        assert semantic_hits[0].chunk_id == "c_agent"
        assert "vector" in semantic_hits[0].sources

        keyword_hits = await retriever.retrieve("scheduler background task", top_k=3)
        assert any(hit.chunk_id == "c_scheduler" and "bm25" in hit.sources for hit in keyword_hits)

        chinese_hits = await retriever.retrieve("工具怎么调用", top_k=3)
        assert any(hit.chunk_id == "c_tool" for hit in chinese_hits)
        assert any("bm25" in hit.sources for hit in chinese_hits)

        citations = build_citations(chinese_hits)
        block, injected_ids = build_knowledge_prompt_block(chinese_hits, citations)
        assert block
        assert "[K1]" in block
        assert injected_ids
    finally:
        store.close()


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


def _hit(
    chunk_id: str,
    *,
    score: float,
    lanes: tuple[str, ...] = ("vector",),
) -> KnowledgeChunkHit:
    return KnowledgeChunkHit(
        chunk_id=chunk_id,
        document_id=f"d_{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        source_path=f"/tmp/{chunk_id}.md",
        title=chunk_id,
        lanes=lanes,
        sources=lanes,
    )


def _seed_knowledge_store(store: KnowledgeStore, tmp_path: Path) -> None:
    docs = [
        (
            "d_agent",
            "agent.md",
            [
                DocumentChunk(
                    id="c_agent",
                    document_id="d_agent",
                    chunk_index=0,
                    text="Agent Runtime orchestrates multi-turn chat, prompt assembly, and the main LLM loop.",
                    content_hash="h_agent",
                )
            ],
            {"c_agent": [1.0, 0.0, 0.0, 0.0]},
        ),
        (
            "d_memory",
            "memory.md",
            [
                DocumentChunk(
                    id="c_memory",
                    document_id="d_memory",
                    chunk_index=0,
                    text="Long-term memory uses retrieval to bring durable user facts into context.",
                    content_hash="h_memory",
                )
            ],
            {"c_memory": [0.0, 1.0, 0.0, 0.0]},
        ),
        (
            "d_tool",
            "tool.md",
            [
                DocumentChunk(
                    id="c_tool",
                    document_id="d_tool",
                    chunk_index=0,
                    text="Tool calling and function calling are registered through the tool system and plugins.",
                    content_hash="h_tool",
                )
            ],
            {"c_tool": [0.0, 0.0, 1.0, 0.0]},
        ),
        (
            "d_scheduler",
            "scheduler.md",
            [
                DocumentChunk(
                    id="c_scheduler",
                    document_id="d_scheduler",
                    chunk_index=0,
                    text="The scheduler runs background task workflows and proactive notification checks.",
                    content_hash="h_scheduler",
                )
            ],
            {"c_scheduler": [0.0, 0.0, 0.0, 1.0]},
        ),
    ]
    for document_id, filename, chunks, vectors in docs:
        store.upsert_document_with_chunks(
            document_id=document_id,
            document=LoadedDocument(
                source_path=tmp_path / filename,
                title=filename,
                content="\n".join(chunk.text for chunk in chunks),
                file_type="markdown",
                content_hash=f"h_{document_id}",
                mtime=0.0,
            ),
            chunks=chunks,
            vectors=vectors,
            embedding_model="keyword-test",
        )


class _KeywordEmbedder:
    async def embed(self, text: str) -> list[float]:
        raw = text.lower()
        return [
            1.0 if "agent" in raw or "runtime" in raw or "turn" in raw else 0.0,
            1.0 if "memory" in raw or "记忆" in raw else 0.0,
            1.0 if "tool" in raw or "function" in raw or "工具" in raw else 0.0,
            1.0 if "scheduler" in raw or "task" in raw or "任务" in raw else 0.0,
        ]
