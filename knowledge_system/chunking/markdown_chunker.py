from __future__ import annotations

import hashlib
import re

from knowledge_system.indexing.models import DocumentChunk, LoadedDocument


class MarkdownChunker:
    def __init__(self, *, max_chars: int = 1200, overlap_chars: int = 120) -> None:
        self._max_chars = max(200, int(max_chars))
        self._overlap_chars = max(0, min(int(overlap_chars), self._max_chars // 2))

    def chunk(self, document_id: str, document: LoadedDocument) -> list[DocumentChunk]:
        sections = self._split_sections(document.content)
        chunks: list[DocumentChunk] = []
        for heading_path, start_line, lines in sections:
            text = "\n".join(lines).strip()
            if not text:
                continue
            pieces = self._split_text(text)
            current_line = start_line
            for piece in pieces:
                line_count = max(1, piece.count("\n") + 1)
                chunk_id = _chunk_id(document_id, len(chunks), piece)
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=piece,
                        heading_path=heading_path,
                        line_start=current_line,
                        line_end=current_line + line_count - 1,
                        token_count=max(1, len(piece) // 4),
                        content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                    )
                )
                current_line += line_count
        if chunks:
            return chunks
        fallback = document.content.strip()
        if not fallback:
            return []
        return [
            DocumentChunk(
                id=_chunk_id(document_id, 0, fallback),
                document_id=document_id,
                chunk_index=0,
                text=fallback[: self._max_chars],
                line_start=1,
                line_end=max(1, fallback.count("\n") + 1),
                token_count=max(1, len(fallback) // 4),
                content_hash=hashlib.sha256(fallback.encode("utf-8")).hexdigest(),
            )
        ]

    def _split_sections(self, content: str) -> list[tuple[str, int, list[str]]]:
        lines = content.splitlines()
        sections: list[tuple[str, int, list[str]]] = []
        current: list[str] = []
        current_start = 1
        heading_stack: list[tuple[int, str]] = []
        current_heading = ""
        for index, line in enumerate(lines, start=1):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match and current:
                sections.append((current_heading, current_start, current))
                current = []
                current_start = index
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack = [item for item in heading_stack if item[0] < level]
                heading_stack.append((level, title))
                current_heading = " > ".join(item[1] for item in heading_stack)
            if not current:
                current_start = index
            current.append(line)
        if current:
            sections.append((current_heading, current_start, current))
        return sections

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > self._max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_text(paragraph))
                continue
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self._max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            current = paragraph
        if current:
            chunks.append(current.strip())
        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self._max_chars)
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - self._overlap_chars, start + 1)
        return [chunk for chunk in chunks if chunk]


def _chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"kchunk_{digest[:20]}"
