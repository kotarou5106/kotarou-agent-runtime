from __future__ import annotations

import hashlib

from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.models import KnowledgeChunkHit
from knowledge_system.indexing.store import KnowledgeStore

_RRF_K = 60
_KEYWORD_RRF_WEIGHT = 0.6


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        embedder: KnowledgeEmbedderAdapter | None = None,
        top_k: int = 6,
        score_threshold: float = 0.20,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = max(1, int(top_k))
        self._score_threshold = float(score_threshold)

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[KnowledgeChunkHit]:
        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        vector_hits: list[KnowledgeChunkHit] = []
        if self._embedder is not None and str(query or "").strip():
            try:
                query_vec = await self._embedder.embed_text(str(query))
                vector_hits = self._store.vector_search(
                    query_vec=query_vec,
                    top_k=actual_top_k,
                    score_threshold=self._score_threshold,
                )
            except Exception:
                vector_hits = []
        keyword_hits = self._store.keyword_search(
            str(query or ""),
            top_k=max(actual_top_k * 2, 12),
        )
        return _rrf_merge(vector_hits, keyword_hits, top_k=actual_top_k)

    def log_event(
        self,
        *,
        query: str,
        trace_id: str = "",
        retrieved_chunk_ids: list[str],
        injected_chunk_ids: list[str],
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
        )


def _rrf_merge(
    vector_hits: list[KnowledgeChunkHit],
    keyword_hits: list[KnowledgeChunkHit],
    *,
    top_k: int,
    k: int = _RRF_K,
) -> list[KnowledgeChunkHit]:
    by_id: dict[str, KnowledgeChunkHit] = {}
    ranks: dict[str, float] = {}
    lanes: dict[str, set[str]] = {}
    for index, hit in enumerate(vector_hits, start=1):
        by_id.setdefault(hit.chunk_id, hit)
        ranks[hit.chunk_id] = ranks.get(hit.chunk_id, 0.0) + 1.0 / (k + index)
        lanes.setdefault(hit.chunk_id, set()).add("vector")
    for index, hit in enumerate(keyword_hits, start=1):
        by_id.setdefault(hit.chunk_id, hit)
        ranks[hit.chunk_id] = ranks.get(hit.chunk_id, 0.0) + _KEYWORD_RRF_WEIGHT / (k + index)
        lanes.setdefault(hit.chunk_id, set()).add("keyword")
    merged: list[KnowledgeChunkHit] = []
    for chunk_id, hit in by_id.items():
        merged.append(
            KnowledgeChunkHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                score=ranks.get(chunk_id, hit.score),
                source_path=hit.source_path,
                title=hit.title,
                heading_path=hit.heading_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
                lanes=tuple(sorted(lanes.get(chunk_id, set(hit.lanes)))),
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]
