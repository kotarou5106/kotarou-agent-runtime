from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    source_path: Path
    title: str
    content: str
    file_type: str
    content_hash: str
    mtime: float


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    heading_path: str = ""
    line_start: int = 0
    line_end: int = 0
    token_count: int = 0
    content_hash: str = ""


@dataclass(frozen=True)
class StoredDocument:
    id: str
    source_path: str
    title: str
    content_hash: str
    mtime: float
    file_type: str
    created_at: str
    updated_at: str
    status: str = "active"


@dataclass(frozen=True)
class KnowledgeCitation:
    id: str
    chunk_id: str
    document_id: str
    source_path: str
    title: str
    heading_path: str = ""
    line_start: int = 0
    line_end: int = 0


@dataclass(frozen=True)
class KnowledgeChunkHit:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_path: str
    title: str
    heading_path: str = ""
    line_start: int = 0
    line_end: int = 0
    lanes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KnowledgeRetrievalEvent:
    id: str
    trace_id: str
    query: str
    retrieved_chunk_ids: list[str]
    injected_chunk_ids: list[str]
    created_at: datetime
