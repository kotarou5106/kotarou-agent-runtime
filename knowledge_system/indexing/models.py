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
    parent_id: str | None = None
    chunk_type: str = "child"
    metadata: dict[str, object] = field(default_factory=dict)


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
    parent_id: str | None = None
    matched_child_ids: tuple[str, ...] = field(default_factory=tuple)


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
    sources: tuple[str, ...] = field(default_factory=tuple)
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    parent_id: str | None = None
    chunk_type: str = "child"
    matched_child_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeRetrievalEvent:
    id: str
    trace_id: str
    query: str
    retrieved_chunk_ids: list[str]
    injected_chunk_ids: list[str]
    created_at: datetime
