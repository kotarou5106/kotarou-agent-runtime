from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_system.chunking import DocumentChunker
from knowledge_system.config import KnowledgeConfig, resolve_knowledge_db_path
from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.indexer import KnowledgeIndexer
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.injection import build_citations, build_knowledge_prompt_block
from knowledge_system.loading import DocumentLoader
from knowledge_system.retrieval import KnowledgeQuery, KnowledgeRetrievalResult, KnowledgeRetriever


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
    ) -> "KnowledgeService":
        cfg = config or KnowledgeConfig()
        store = KnowledgeStore(resolve_knowledge_db_path(workspace, cfg.db_path))
        adapter = KnowledgeEmbedderAdapter(embedder) if embedder is not None else None
        loader = DocumentLoader()
        chunker = DocumentChunker(
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )
        indexer = KnowledgeIndexer(
            store=store,
            loader=loader,
            chunker=chunker,
            embedder=adapter,
            embedding_model=embedding_model,
        )
        retriever = KnowledgeRetriever(
            store=store,
            embedder=adapter,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
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
        hits = await self.retriever.retrieve(query.text, top_k=query.top_k)
        citations = build_citations(hits)
        block, injected_ids = build_knowledge_prompt_block(
            hits,
            citations,
            max_chars=self._inject_max_chars,
        )
        retrieved_ids = [hit.chunk_id for hit in hits]
        if retrieved_ids or injected_ids:
            self.retriever.log_event(
                query=query.text,
                trace_id=query.trace_id,
                retrieved_chunk_ids=retrieved_ids,
                injected_chunk_ids=injected_ids,
            )
        injected = set(injected_ids)
        return KnowledgeRetrievalResult(
            block=block,
            hits=hits,
            citations=[
                citation for citation in citations if citation.chunk_id in injected
            ],
            trace={
                "retrieved_count": len(hits),
                "injected_count": len(injected_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "injected_chunk_ids": injected_ids,
            },
        )

    def close(self) -> None:
        self.store.close()
