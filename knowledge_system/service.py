from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_system.chunking import DocumentChunker
from knowledge_system.config import KnowledgeConfig, resolve_knowledge_db_path
from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.indexer import KnowledgeIndexer
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.injection import (
    build_citations,
    build_knowledge_prompt_block,
    validate_answer_citations,
)
from knowledge_system.loading import DocumentLoader
from knowledge_system.retrieval import KnowledgeQuery, KnowledgeRetrievalResult, KnowledgeRetriever
from knowledge_system.retrieval.reranker import LLMReranker
from knowledge_system.retrieval.trace import KnowledgeRetrieverResult, hits_to_trace_hits
from knowledge_system.retrieval.trace_formatter import (
    citation_blocks_to_trace,
    citation_validation_to_trace,
)


class KnowledgeService:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        indexer: KnowledgeIndexer,
        retriever: KnowledgeRetriever,
        inject_max_chars: int = 2400,
    ) -> None:
        self.store = store
        self.indexer = indexer
        self.retriever = retriever
        self._inject_max_chars = max(400, int(inject_max_chars))

    @classmethod
    def create(
        cls,
        *,
        workspace: Path,
        config: KnowledgeConfig | None = None,
        embedder: Any | None = None,
        embedding_model: str = "",
        reranker_provider: Any | None = None,
        reranker_model: str = "",
    ) -> "KnowledgeService":
        cfg = config or KnowledgeConfig()
        store = KnowledgeStore(resolve_knowledge_db_path(workspace, cfg.db_path))
        adapter = KnowledgeEmbedderAdapter(embedder) if embedder is not None else None
        loader = DocumentLoader()
        chunker = DocumentChunker(
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
            parent_child_enabled=cfg.parent_child.enabled,
            parent_max_chars=cfg.parent_child.parent_chunk_size,
            parent_overlap_chars=cfg.parent_child.parent_chunk_overlap,
            child_max_chars=cfg.parent_child.child_chunk_size,
            child_overlap_chars=cfg.parent_child.child_chunk_overlap,
        )
        indexer = KnowledgeIndexer(
            store=store,
            loader=loader,
            chunker=chunker,
            embedder=adapter,
            embedding_model=embedding_model,
        )
        reranker = None
        reranking_enabled = cfg.reranking.enabled and cfg.reranking.type == "llm"
        if reranking_enabled and reranker_provider is not None and reranker_model:
            reranker = LLMReranker(
                reranker_provider,
                model=reranker_model,
                top_n=cfg.reranking.top_n,
                timeout_seconds=cfg.reranking.timeout_seconds,
            )
        else:
            reranking_enabled = False
        retriever = KnowledgeRetriever(
            store=store,
            embedder=adapter,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
            reranker=reranker,
            reranking_enabled=reranking_enabled,
            parent_child_enabled=cfg.parent_child.enabled,
            expand_to_parent=cfg.parent_child.expand_to_parent,
        )
        return cls(
            store=store,
            indexer=indexer,
            retriever=retriever,
            inject_max_chars=cfg.inject_max_chars,
        )

    async def index_path(self, path: str | Path, *, force: bool = False) -> dict[str, object]:
        return await self.indexer.index_path(path, force=force)

    async def retrieve(self, query: KnowledgeQuery) -> KnowledgeRetrievalResult:
        retrieval = await self.retriever.retrieve(
            query.text,
            top_k=query.top_k,
            include_trace=True,
            trace_id=query.trace_id or None,
        )
        if isinstance(retrieval, KnowledgeRetrieverResult):
            hits = retrieval.hits
            trace = retrieval.trace
        else:
            hits = retrieval
            trace = None

        if trace is not None:
            started_citation = self._start_stage(trace, "citation")
        citations = build_citations(hits)
        if trace is not None:
            trace.citation_ids = [citation.id for citation in citations]
            trace.citation_blocks = citation_blocks_to_trace(citations)
            self._finish_stage(trace, "citation", started_citation)

        if trace is not None:
            started_injection = self._start_stage(trace, "prompt_injection")
        block, injected_ids = build_knowledge_prompt_block(
            hits,
            citations,
            max_chars=self._inject_max_chars,
        )
        if trace is not None:
            self._finish_stage(trace, "prompt_injection", started_injection)
        retrieved_ids = [hit.chunk_id for hit in hits]
        injected = set(injected_ids)
        injected_hits = [hit for hit in hits if hit.chunk_id in injected]
        if trace is not None:
            started_citation_validation = self._start_stage(trace, "citation_validation")
            trace.selected_chunks = hits_to_trace_hits(
                injected_hits,
                source="prompt_injection",
            )
            validation = validate_answer_citations(block, citations)
            trace.citation_validation = citation_validation_to_trace(validation, citations)
            self._finish_stage(
                trace,
                "citation_validation",
                started_citation_validation,
            )
            if not trace.citation_validation.is_valid:
                trace.warnings.append(
                    trace.citation_validation.message or "citation validation failed"
                )
            if not injected_hits and hits:
                trace.warnings.append("retrieved hits were not injected because of prompt budget")
        if retrieved_ids or injected_ids:
            event_trace_id = query.trace_id or (trace.trace_id if trace is not None else "")
            self.retriever.log_event(
                query=query.text,
                trace_id=event_trace_id,
                retrieved_chunk_ids=retrieved_ids,
                injected_chunk_ids=injected_ids,
                trace=trace.to_dict() if trace is not None else None,
            )
        trace_dict = trace.to_dict() if trace is not None else {}
        trace_dict.update(
            {
                "retrieved_count": len(hits),
                "injected_count": len(injected_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "injected_chunk_ids": injected_ids,
            }
        )
        return KnowledgeRetrievalResult(
            block=block,
            hits=hits,
            citations=[
                citation for citation in citations if citation.chunk_id in injected
            ],
            trace=trace_dict,
        )

    def close(self) -> None:
        self.store.close()

    def _start_stage(self, trace: Any, name: str) -> float:
        import time

        return time.perf_counter()

    def _finish_stage(self, trace: Any, name: str, started: float) -> None:
        import time

        elapsed = round((time.perf_counter() - started) * 1000.0, 3)
        trace.stage_timings_ms[name] = trace.stage_timings_ms.get(name, 0.0) + elapsed
