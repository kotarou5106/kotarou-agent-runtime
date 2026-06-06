from __future__ import annotations

import hashlib
from dataclasses import replace

from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.models import KnowledgeChunkHit
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.retrieval.parent_child import expand_hits_to_parents
from knowledge_system.retrieval.query_rewrite import rewrite_query
from knowledge_system.retrieval.reranker import BaseReranker, NoOpReranker
from knowledge_system.retrieval.trace import KnowledgeRetrieverResult, KnowledgeTraceCollector

_RRF_K = 60


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        embedder: KnowledgeEmbedderAdapter | None = None,
        top_k: int = 6,
        score_threshold: float = 0.20,
        reranker: BaseReranker | None = None,
        reranking_enabled: bool = False,
        parent_child_enabled: bool = True,
        expand_to_parent: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = max(1, int(top_k))
        self._score_threshold = float(score_threshold)
        self._reranker = reranker or NoOpReranker()
        self._reranking_enabled = bool(reranking_enabled)
        self._parent_child_enabled = bool(parent_child_enabled)
        self._expand_to_parent = bool(expand_to_parent)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_trace: bool = False,
        trace_id: str | None = None,
        retrieval_mode: str = "hybrid_rrf",
    ) -> list[KnowledgeChunkHit] | KnowledgeRetrieverResult:
        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        mode = _normalize_mode(retrieval_mode)
        collector = KnowledgeTraceCollector(trace_id=trace_id) if include_trace else None
        if collector is not None:
            collector.start(query)
        with collector.stage("query_rewrite") if collector else _null_stage():
            variants = rewrite_query(query) if mode == "hybrid_rrf" else _single_query(query)
        if collector is not None:
            collector.record_query_variants(variants)
        if not variants:
            hits: list[KnowledgeChunkHit] = []
            return KnowledgeRetrieverResult(hits=hits, trace=collector.finish()) if collector else hits

        ranked_lists: list[tuple[str, list[KnowledgeChunkHit]]] = []
        for variant in variants:
            if mode in {"vector", "hybrid_rrf"}:
                with collector.stage("vector_retrieval") if collector else _null_stage():
                    vector_hits = await self._vector_retrieve(
                        variant,
                        top_k=actual_top_k,
                        collector=collector,
                    )
                if collector is not None:
                    collector.record_vector_hits(variant, vector_hits)
                if vector_hits:
                    ranked_lists.append(("vector", vector_hits))

            if mode in {"bm25", "hybrid_rrf"}:
                with collector.stage("bm25_retrieval") if collector else _null_stage():
                    bm25_hits = self._store.keyword_search(
                        variant,
                        top_k=max(actual_top_k * 2, 12),
                    )
                if collector is not None:
                    collector.record_bm25_hits(variant, bm25_hits)
                if bm25_hits:
                    ranked_lists.append(("bm25", bm25_hits))

        with collector.stage("rrf_merge") if collector else _null_stage():
            if mode == "vector":
                hits = ranked_lists[0][1][:actual_top_k] if ranked_lists else []
            elif mode == "bm25":
                hits = ranked_lists[0][1][:actual_top_k] if ranked_lists else []
            else:
                hits = rrf_merge(ranked_lists, top_k=actual_top_k)
        if collector is not None:
            collector.record_rrf_hits(hits)

        rerank_input = hits
        with collector.stage("reranking") if collector else _null_stage():
            if self._reranking_enabled and hits:
                hits = await self._reranker.rerank(query, hits, top_k=actual_top_k)
                if not hits:
                    hits = rerank_input
            else:
                hits = hits[:actual_top_k]
        if collector is not None:
            collector.record_reranking(
                enabled=self._reranking_enabled,
                skipped=not self._reranking_enabled,
                skip_reason="" if self._reranking_enabled else "disabled",
                input_hits=rerank_input,
                output_hits=hits,
                warning=str(getattr(self._reranker, "last_warning", "") or ""),
            )

        parent_input = hits
        with collector.stage("parent_child_expansion") if collector else _null_stage():
            hits, parent_trace = expand_hits_to_parents(
                hits,
                store=self._store,
                enabled=self._parent_child_enabled and self._expand_to_parent,
            )
            hits = hits[:actual_top_k]
        if collector is not None:
            collector.record_parent_child_expansion(
                enabled=self._parent_child_enabled and self._expand_to_parent,
                input_hits=parent_input,
                output_hits=hits,
                metadata=parent_trace,
            )

        if collector is not None:
            with collector.stage("context_selection"):
                collector.record_selected_chunks(hits)
            if not hits:
                collector.record_warning("no retrieval hits")
            return KnowledgeRetrieverResult(hits=hits, trace=collector.finish())
        return hits

    async def _vector_retrieve(
        self,
        query: str,
        *,
        top_k: int,
        collector: KnowledgeTraceCollector | None = None,
    ) -> list[KnowledgeChunkHit]:
        if self._embedder is None or not str(query or "").strip():
            return []
        try:
            query_vec = await self._embedder.embed_text(str(query))
            hits = self._store.vector_search(
                query_vec=query_vec,
                top_k=top_k,
                score_threshold=self._score_threshold,
            )
        except Exception as exc:
            if collector is not None:
                collector.record_warning(f"embedding/vector retrieval failed; BM25 fallback may still succeed: {exc}")
            return []
        return [_with_source(hit, "vector") for hit in hits]

    def log_event(
        self,
        *,
        query: str,
        trace_id: str = "",
        retrieved_chunk_ids: list[str],
        injected_chunk_ids: list[str],
        trace: dict[str, object] | None = None,
    ) -> None:
        digest = hashlib.sha1(
            f"{trace_id}:{query}:{','.join(retrieved_chunk_ids)}".encode("utf-8")
        ).hexdigest()
        self._store.log_retrieval_event(
            event_id=f"kretr_{digest[:24]}",
            trace_id=trace_id,
            query=query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            injected_chunk_ids=injected_chunk_ids,
            trace=trace,
        )


def rrf_merge(
    ranked_hit_lists: list[tuple[str, list[KnowledgeChunkHit]]],
    *,
    top_k: int,
    k: int = _RRF_K,
) -> list[KnowledgeChunkHit]:
    """Merge multiple ranked retrieval lanes with Reciprocal Rank Fusion."""
    if not ranked_hit_lists:
        return []

    by_id: dict[str, KnowledgeChunkHit] = {}
    best_original_score: dict[str, float] = {}
    rrf_scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    vector_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for source_name, hits in ranked_hit_lists:
        normalized_source = _normalize_source(source_name)
        seen_in_lane: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if not hit.chunk_id or hit.chunk_id in seen_in_lane:
                continue
            seen_in_lane.add(hit.chunk_id)
            existing_score = best_original_score.get(hit.chunk_id, float("-inf"))
            if hit.chunk_id not in by_id or hit.score > existing_score:
                by_id[hit.chunk_id] = hit
                best_original_score[hit.chunk_id] = hit.score
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            source_set = sources.setdefault(hit.chunk_id, set())
            source_set.add(normalized_source)
            source_set.update(_normalize_source(source) for source in hit.sources)
            source_set.update(_normalize_source(lane) for lane in hit.lanes)
            if normalized_source == "vector":
                vector_ranks[hit.chunk_id] = min(vector_ranks.get(hit.chunk_id, rank), rank)
            if normalized_source == "bm25":
                bm25_ranks[hit.chunk_id] = min(bm25_ranks.get(hit.chunk_id, rank), rank)

    merged: list[KnowledgeChunkHit] = []
    for chunk_id, hit in by_id.items():
        hit_sources = tuple(sorted(source for source in sources.get(chunk_id, set()) if source))
        merged.append(
            replace(
                hit,
                score=rrf_scores.get(chunk_id, hit.score),
                lanes=hit_sources,
                sources=hit_sources,
                vector_rank=vector_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
                rrf_score=rrf_scores.get(chunk_id),
            )
        )
    merged.sort(
        key=lambda item: (
            item.rrf_score if item.rrf_score is not None else item.score,
            best_original_score.get(item.chunk_id, item.score),
        ),
        reverse=True,
    )
    return merged[: max(1, int(top_k))]


def _rrf_merge(
    vector_hits: list[KnowledgeChunkHit],
    keyword_hits: list[KnowledgeChunkHit],
    *,
    top_k: int,
    k: int = _RRF_K,
) -> list[KnowledgeChunkHit]:
    """Compatibility wrapper for older tests and callers."""
    return rrf_merge(
        [("vector", vector_hits), ("bm25", keyword_hits)],
        top_k=top_k,
        k=k,
    )


def _with_source(hit: KnowledgeChunkHit, source: str) -> KnowledgeChunkHit:
    normalized = _normalize_source(source)
    lanes = tuple(sorted({*hit.lanes, normalized}))
    sources = tuple(sorted({*hit.sources, normalized}))
    return replace(hit, lanes=lanes, sources=sources)


def _normalize_source(source: str) -> str:
    value = str(source or "").strip().lower()
    if value == "keyword":
        return "bm25"
    if value == "sqlite-vec":
        return "vector"
    return value


def _normalize_mode(mode: str) -> str:
    value = str(mode or "hybrid_rrf").strip().lower()
    if value in {"hybrid", "rrf", "hybrid_rrf"}:
        return "hybrid_rrf"
    if value in {"bm25", "keyword", "sparse"}:
        return "bm25"
    if value in {"vector", "dense"}:
        return "vector"
    return "hybrid_rrf"


def _single_query(query: str) -> list[str]:
    text = str(query or "").strip()
    return [text] if text else []


class _null_stage:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False
