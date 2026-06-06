from __future__ import annotations

import hashlib

from knowledge_system.chunking.markdown_chunker import MarkdownChunker
from knowledge_system.indexing.models import DocumentChunk, LoadedDocument


class DocumentChunker:
    def __init__(
        self,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 120,
        parent_child_enabled: bool = False,
        parent_max_chars: int | None = None,
        parent_overlap_chars: int | None = None,
        child_max_chars: int | None = None,
        child_overlap_chars: int | None = None,
    ) -> None:
        self._parent_child_enabled = bool(parent_child_enabled)
        self._markdown = MarkdownChunker(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        self._parent_markdown = MarkdownChunker(
            max_chars=parent_max_chars or max_chars,
            overlap_chars=parent_overlap_chars if parent_overlap_chars is not None else overlap_chars,
        )
        self._child_markdown = MarkdownChunker(
            max_chars=child_max_chars or max(200, max_chars // 2),
            overlap_chars=child_overlap_chars if child_overlap_chars is not None else max(40, overlap_chars // 2),
        )

    def chunk(self, document_id: str, document: LoadedDocument) -> list[DocumentChunk]:
        if not self._parent_child_enabled:
            return self._markdown.chunk(document_id, document)
        return self._parent_child_chunks(document_id, document)

    def _parent_child_chunks(
        self,
        document_id: str,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        parents = [
            _as_parent_chunk(chunk, index)
            for index, chunk in enumerate(self._parent_markdown.chunk(document_id, document))
        ]
        if not parents:
            return []
        chunks: list[DocumentChunk] = []
        child_index = 0
        for parent in parents:
            chunks.append(parent)
            parent_doc = LoadedDocument(
                source_path=document.source_path,
                title=document.title,
                content=parent.text,
                file_type=document.file_type,
                content_hash=parent.content_hash,
                mtime=document.mtime,
            )
            children = self._child_markdown.chunk(document_id, parent_doc)
            for child in children:
                child_id = _child_chunk_id(parent.id, child_index, child.text)
                chunks.append(
                    DocumentChunk(
                        id=child_id,
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=child.text,
                        heading_path=child.heading_path or parent.heading_path,
                        line_start=parent.line_start,
                        line_end=parent.line_end,
                        token_count=child.token_count,
                        content_hash=child.content_hash,
                        parent_id=parent.id,
                        chunk_type="child",
                        metadata={"parent_chunk_index": parent.chunk_index},
                    )
                )
                child_index += 1
        return chunks


def _as_parent_chunk(chunk: DocumentChunk, index: int) -> DocumentChunk:
    return DocumentChunk(
        id=f"{chunk.id}_parent",
        document_id=chunk.document_id,
        chunk_index=index,
        text=chunk.text,
        heading_path=chunk.heading_path,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        token_count=chunk.token_count,
        content_hash=chunk.content_hash,
        parent_id=None,
        chunk_type="parent",
        metadata=dict(chunk.metadata),
    )


def _child_chunk_id(parent_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{parent_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"kchild_{digest[:20]}"
