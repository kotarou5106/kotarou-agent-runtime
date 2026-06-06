from __future__ import annotations

import hashlib
from pathlib import Path

from knowledge_system.chunking import DocumentChunker
from knowledge_system.embedding import KnowledgeEmbedderAdapter
from knowledge_system.indexing.models import LoadedDocument
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.loading import DocumentLoader


class KnowledgeIndexer:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        embedder: KnowledgeEmbedderAdapter | None = None,
        embedding_model: str = "",
    ) -> None:
        self._store = store
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._embedding_model = embedding_model

    async def index_path(self, path: str | Path, *, force: bool = False) -> dict[str, object]:
        document = self._loader.load(path)
        document_id = _document_id(document)
        existing = self._store.get_document_by_source_path(document.source_path)
        if (
            not force
            and existing is not None
            and str(existing.get("content_hash") or "") == document.content_hash
        ):
            return {
                "document_id": document_id,
                "source_path": str(document.source_path),
                "skipped": True,
                "chunk_count": 0,
            }

        chunks = self._chunker.chunk(document_id, document)
        vectors: dict[str, list[float]] = {}
        searchable_chunks = [
            chunk for chunk in chunks if str(chunk.chunk_type or "child") != "parent"
        ]
        if self._embedder is not None and searchable_chunks:
            embedded = await self._embedder.embed_texts([chunk.text for chunk in searchable_chunks])
            vectors = {
                chunk.id: vector
                for chunk, vector in zip(searchable_chunks, embedded)
                if vector
            }
        self._store.upsert_document_with_chunks(
            document_id=document_id,
            document=document,
            chunks=chunks,
            vectors=vectors,
            embedding_model=self._embedding_model,
        )
        return {
            "document_id": document_id,
            "source_path": str(document.source_path),
            "skipped": False,
            "chunk_count": len(chunks),
            "vector_count": len(vectors),
        }


def _document_id(document: LoadedDocument) -> str:
    source = str(document.source_path.expanduser().resolve())
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return f"kdoc_{digest[:20]}"
