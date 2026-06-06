from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_system.chunking import DocumentChunker
from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.indexer import KnowledgeIndexer
from knowledge_system.indexing.models import DocumentChunk, KnowledgeChunkHit, LoadedDocument
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.injection import build_citations, build_knowledge_prompt_block
from knowledge_system.loading import DocumentLoader
from knowledge_system.retrieval.parent_child import expand_hits_to_parents
from knowledge_system.retrieval.reranker import LLMReranker, NoOpReranker
from knowledge_system.retrieval.retriever import KnowledgeRetriever
from knowledge_system.retrieval.trace import KnowledgeRetrieverResult


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order() -> None:
    hits = [_hit("c1", score=0.8), _hit("c2", score=0.7)]

    result = await NoOpReranker().rerank("query", hits, top_k=2)

    assert [hit.chunk_id for hit in result] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_llm_reranker_reorders_and_writes_score() -> None:
    reranker = LLMReranker(
        _MockProvider('{"scores":[{"id":"c2","score":0.95},{"id":"c1","score":0.1}]}'),
        model="mock",
    )

    result = await reranker.rerank("tool", [_hit("c1", score=0.9), _hit("c2", score=0.2)], top_k=2)

    assert [hit.chunk_id for hit in result] == ["c2", "c1"]
    assert result[0].rerank_score == 0.95
    assert result[0].metadata["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_llm_reranker_falls_back_on_bad_json_or_exception() -> None:
    hits = [_hit("c1", score=0.9), _hit("c2", score=0.2)]
    bad = await LLMReranker(_MockProvider("not-json"), model="mock").rerank("q", hits, top_k=2)
    boom = await LLMReranker(_BoomProvider(), model="mock").rerank("q", hits, top_k=2)

    assert [hit.chunk_id for hit in bad] == ["c1", "c2"]
    assert [hit.chunk_id for hit in boom] == ["c1", "c2"]
    assert bad
    assert boom


def test_parent_child_expansion_dedupes_and_aggregates(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_parent_child(store, tmp_path)
        hits = [
            _hit("child_a", score=0.4, parent_id="parent_1"),
            _hit("child_b", score=0.9, parent_id="parent_1"),
        ]

        expanded, trace = expand_hits_to_parents(hits, store=store, enabled=True)

        assert len(expanded) == 1
        assert expanded[0].chunk_id == "parent_1"
        assert expanded[0].score == 0.9
        assert expanded[0].matched_child_ids == ("child_a", "child_b")
        assert expanded[0].metadata["matched_child_ids"] == ["child_a", "child_b"]
        assert trace["child_to_parent"] == {"child_a": "parent_1", "child_b": "parent_1"}
    finally:
        store.close()


def test_parent_child_expansion_falls_back_for_old_data(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        hit = _hit("legacy", score=0.5, parent_id=None)
        expanded, trace = expand_hits_to_parents([hit], store=store, enabled=True)

        assert expanded[0].chunk_id == "legacy"
        assert expanded[0].matched_child_ids == ("legacy",)
        assert trace["fallback_count"] == 1
    finally:
        store.close()


def test_store_saves_parent_and_child_chunks(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_parent_child(store, tmp_path)
        parent = store.get_chunk("parent_1")
        child_parent = store.get_parent_for_child("child_a")
        chunks = store.list_chunks(document_id="doc_parent", limit=10)

        assert parent is not None
        assert parent.chunk_type == "parent"
        assert child_parent is not None
        assert child_parent.chunk_id == "parent_1"
        assert {item["id"] for item in chunks} >= {"parent_1", "child_a", "child_b"}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_pipeline_rerank_and_parent_child_trace(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_parent_child(store, tmp_path, vectors=True)
        retriever = KnowledgeRetriever(
            store=store,
            embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
            top_k=3,
            score_threshold=0.01,
            reranker=LLMReranker(_MockProvider('{"scores":[{"id":"child_b","score":1.0},{"id":"child_a","score":0.2}]}'), model="mock"),
            reranking_enabled=True,
            parent_child_enabled=True,
            expand_to_parent=True,
        )

        result = await retriever.retrieve("tool calling", include_trace=True)

        assert isinstance(result, KnowledgeRetrieverResult)
        assert result.hits[0].chunk_id == "parent_1"
        assert result.trace.reranking["enabled"] is True
        assert result.trace.parent_child_expansion["enabled"] is True
        assert result.trace.parent_child_expansion["parent_hits"]

        citations = build_citations(result.hits)
        block, _ = build_knowledge_prompt_block(result.hits, citations)
        assert "PARENT CONTEXT" in block
        assert "tiny child" not in block
    finally:
        store.close()


@pytest.mark.asyncio
async def test_pipeline_disabled_rerank_and_parent_child_modes(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _seed_parent_child(store, tmp_path, vectors=True)
        retriever = KnowledgeRetriever(
            store=store,
            embedder=KnowledgeEmbedderAdapter(_KeywordEmbedder()),
            reranking_enabled=False,
            parent_child_enabled=False,
        )

        result = await retriever.retrieve("", include_trace=True)
        non_empty = await retriever.retrieve("tool", include_trace=True)

        assert isinstance(result, KnowledgeRetrieverResult)
        assert result.hits == []
        assert isinstance(non_empty, KnowledgeRetrieverResult)
        assert non_empty.trace.reranking["skipped"] is True
        assert non_empty.trace.parent_child_expansion["enabled"] is False
    finally:
        store.close()


def test_parent_child_chunker_creates_parent_and_children(tmp_path: Path) -> None:
    document = LoadedDocument(
        source_path=tmp_path / "doc.md",
        title="doc",
        content="Alpha tool calling paragraph.\n\nBeta memory paragraph.",
        file_type="markdown",
        content_hash="h",
        mtime=0.0,
    )
    chunks = DocumentChunker(
        parent_child_enabled=True,
        parent_max_chars=1200,
        child_max_chars=220,
    ).chunk("doc", document)

    assert any(chunk.chunk_type == "parent" for chunk in chunks)
    assert any(chunk.chunk_type == "child" and chunk.parent_id for chunk in chunks)


def _hit(
    chunk_id: str,
    *,
    score: float,
    parent_id: str | None = None,
) -> KnowledgeChunkHit:
    return KnowledgeChunkHit(
        chunk_id=chunk_id,
        document_id="doc_parent",
        text=f"tiny child {chunk_id}",
        score=score,
        source_path="/tmp/doc.md",
        title="doc",
        parent_id=parent_id,
        chunk_type="child",
        lanes=("bm25",),
        sources=("bm25",),
    )


def _seed_parent_child(store: KnowledgeStore, tmp_path: Path, *, vectors: bool = False) -> None:
    document = LoadedDocument(
        source_path=tmp_path / "doc.md",
        title="doc",
        content="PARENT CONTEXT about tool calling, plugins, and lifecycle hooks.",
        file_type="markdown",
        content_hash="h_doc",
        mtime=0.0,
    )
    chunks = [
        DocumentChunk(
            id="parent_1",
            document_id="doc_parent",
            chunk_index=0,
            text="PARENT CONTEXT about tool calling, plugins, and lifecycle hooks.",
            content_hash="h_parent",
            chunk_type="parent",
        ),
        DocumentChunk(
            id="child_a",
            document_id="doc_parent",
            chunk_index=1,
            text="tiny child about plugins",
            content_hash="h_child_a",
            parent_id="parent_1",
            chunk_type="child",
        ),
        DocumentChunk(
            id="child_b",
            document_id="doc_parent",
            chunk_index=2,
            text="tiny child about tool calling",
            content_hash="h_child_b",
            parent_id="parent_1",
            chunk_type="child",
        ),
    ]
    store.upsert_document_with_chunks(
        document_id="doc_parent",
        document=document,
        chunks=chunks,
        vectors={"child_a": [1.0], "child_b": [1.0]} if vectors else {},
        embedding_model="test",
    )


class _MockProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    async def chat(self, **kwargs):
        return SimpleNamespace(content=self.content)


class _BoomProvider:
    async def chat(self, **kwargs):
        raise RuntimeError("boom")


class _KeywordEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0 if "tool" in text.lower() else 0.0]
