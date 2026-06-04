from __future__ import annotations

from knowledge_system.chunking.markdown_chunker import MarkdownChunker
from knowledge_system.indexing.models import DocumentChunk, LoadedDocument


class DocumentChunker:
    def __init__(
        self,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 120,
    ) -> None:
        self._markdown = MarkdownChunker(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    def chunk(self, document_id: str, document: LoadedDocument) -> list[DocumentChunk]:
        return self._markdown.chunk(document_id, document)
