from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from knowledge_system.indexing.models import KnowledgeChunkHit

_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class TraceHit:
    chunk_id: str
    document_id: str | None = None
    title: str | None = None
    text_preview: str = ""
    score: float | None = None
    rank: int | None = None
    source: str = ""
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitationValidationTrace:
    is_valid: bool
    missing_citations: list[str] = field(default_factory=list)
    unused_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeRetrievalTrace:
    trace_id: str
    original_query: str
    query_variants: list[str] = field(default_factory=list)
    vector_hits_by_query: dict[str, list[TraceHit]] = field(default_factory=dict)
    bm25_hits_by_query: dict[str, list[TraceHit]] = field(default_factory=dict)
    rrf_merged_hits: list[TraceHit] = field(default_factory=list)
    reranking: dict[str, Any] = field(default_factory=dict)
    parent_child_expansion: dict[str, Any] = field(default_factory=dict)
    selected_chunks: list[TraceHit] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    citation_blocks: list[dict[str, Any]] = field(default_factory=list)
    citation_validation: CitationValidationTrace | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "original_query": self.original_query,
            "query_variants": list(self.query_variants),
            "vector_hits_by_query": {
                query: [hit.to_dict() for hit in hits]
                for query, hits in self.vector_hits_by_query.items()
            },
            "bm25_hits_by_query": {
                query: [hit.to_dict() for hit in hits]
                for query, hits in self.bm25_hits_by_query.items()
            },
            "rrf_merged_hits": [hit.to_dict() for hit in self.rrf_merged_hits],
            "reranking": dict(self.reranking),
            "parent_child_expansion": dict(self.parent_child_expansion),
            "selected_chunks": [hit.to_dict() for hit in self.selected_chunks],
            "citation_ids": list(self.citation_ids),
            "citation_blocks": list(self.citation_blocks),
            "citation_validation": (
                self.citation_validation.to_dict() if self.citation_validation else None
            ),
            "stage_timings_ms": dict(self.stage_timings_ms),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class KnowledgeRetrieverResult:
    hits: list[KnowledgeChunkHit]
    trace: KnowledgeRetrievalTrace


class KnowledgeTraceCollector:
    """Collects lightweight per-stage Knowledge RAG trace data."""

    def __init__(self, *, trace_id: str | None = None, preview_chars: int = _PREVIEW_CHARS) -> None:
        self._preview_chars = max(40, int(preview_chars))
        self.trace_id = trace_id or f"ktrace_{uuid.uuid4().hex[:24]}"
        self.trace: KnowledgeRetrievalTrace | None = None

    def start(self, original_query: str) -> KnowledgeRetrievalTrace:
        self.trace = KnowledgeRetrievalTrace(
            trace_id=self.trace_id,
            original_query=str(original_query or ""),
        )
        if not str(original_query or "").strip():
            self.record_warning("empty query")
        return self.trace

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.record_error(f"{name} failed: {exc}")
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._add_timing(name, elapsed)

    def record_query_variants(self, variants: list[str]) -> None:
        try:
            if self.trace is not None:
                self.trace.query_variants = list(variants)
        except Exception:
            pass

    def record_vector_hits(self, query: str, hits: list[KnowledgeChunkHit]) -> None:
        self._record_hits("vector", self._ensure_query(query), hits)
        if not hits:
            self.record_warning(f"no vector hits for query: {query}")

    def record_bm25_hits(self, query: str, hits: list[KnowledgeChunkHit]) -> None:
        self._record_hits("bm25", self._ensure_query(query), hits)
        if not hits:
            self.record_warning(f"no BM25 hits for query: {query}")

    def record_rrf_hits(self, hits: list[KnowledgeChunkHit]) -> None:
        try:
            if self.trace is not None:
                self.trace.rrf_merged_hits = hits_to_trace_hits(
                    hits,
                    source="rrf",
                    preview_chars=self._preview_chars,
                )
        except Exception:
            pass

    def record_selected_chunks(self, hits: list[KnowledgeChunkHit]) -> None:
        try:
            if self.trace is not None:
                self.trace.selected_chunks = hits_to_trace_hits(
                    hits,
                    source="selected",
                    preview_chars=self._preview_chars,
                )
        except Exception:
            pass

    def record_reranking(
        self,
        *,
        enabled: bool,
        skipped: bool,
        input_hits: list[KnowledgeChunkHit],
        output_hits: list[KnowledgeChunkHit],
        skip_reason: str = "",
        warning: str = "",
    ) -> None:
        try:
            if self.trace is not None:
                self.trace.reranking = {
                    "enabled": enabled,
                    "skipped": skipped,
                    "skip_reason": skip_reason,
                    "input_hits": [hit.to_dict() for hit in hits_to_trace_hits(input_hits, source="rerank_input", preview_chars=self._preview_chars)],
                    "output_hits": [hit.to_dict() for hit in hits_to_trace_hits(output_hits, source="rerank_output", preview_chars=self._preview_chars)],
                }
                if warning:
                    self.record_warning(warning)
        except Exception:
            pass

    def record_parent_child_expansion(
        self,
        *,
        enabled: bool,
        input_hits: list[KnowledgeChunkHit],
        output_hits: list[KnowledgeChunkHit],
        metadata: dict[str, Any],
    ) -> None:
        try:
            if self.trace is not None:
                self.trace.parent_child_expansion = {
                    "enabled": enabled,
                    **dict(metadata),
                    "input_hits": [hit.to_dict() for hit in hits_to_trace_hits(input_hits, source="parent_child_input", preview_chars=self._preview_chars)],
                    "parent_hits": [hit.to_dict() for hit in hits_to_trace_hits(output_hits, source="parent_child_output", preview_chars=self._preview_chars)],
                }
                if metadata.get("fallback_count"):
                    self.record_warning("parent_child_fallback")
        except Exception:
            pass

    def record_citation_ids(self, citation_ids: list[str]) -> None:
        try:
            if self.trace is not None:
                self.trace.citation_ids = list(citation_ids)
        except Exception:
            pass

    def record_citation_blocks(self, blocks: list[dict[str, Any]]) -> None:
        try:
            if self.trace is not None:
                self.trace.citation_blocks = list(blocks)
        except Exception:
            pass

    def record_citation_validation(self, validation: CitationValidationTrace) -> None:
        try:
            if self.trace is not None:
                self.trace.citation_validation = validation
                if not validation.is_valid:
                    self.record_warning(validation.message or "citation validation failed")
        except Exception:
            pass

    def record_warning(self, warning: str) -> None:
        try:
            if self.trace is not None:
                text = str(warning or "").strip()
                if text and text not in self.trace.warnings:
                    self.trace.warnings.append(text)
        except Exception:
            pass

    def record_error(self, error: str) -> None:
        try:
            if self.trace is not None:
                text = str(error or "").strip()
                if text and text not in self.trace.errors:
                    self.trace.errors.append(text)
        except Exception:
            pass

    def finish(self) -> KnowledgeRetrievalTrace:
        if self.trace is None:
            self.start("")
        return self.trace

    def _record_hits(self, source: str, query: str, hits: list[KnowledgeChunkHit]) -> None:
        try:
            if self.trace is None:
                return
            trace_hits = hits_to_trace_hits(
                hits,
                source=source,
                preview_chars=self._preview_chars,
            )
            if source == "vector":
                self.trace.vector_hits_by_query[query] = trace_hits
            elif source == "bm25":
                self.trace.bm25_hits_by_query[query] = trace_hits
        except Exception:
            pass

    def _add_timing(self, name: str, elapsed_ms: float) -> None:
        try:
            if self.trace is not None:
                self.trace.stage_timings_ms[name] = (
                    self.trace.stage_timings_ms.get(name, 0.0) + round(elapsed_ms, 3)
                )
        except Exception:
            pass

    def _ensure_query(self, query: str) -> str:
        return str(query or "")


def hits_to_trace_hits(
    hits: list[KnowledgeChunkHit],
    *,
    source: str,
    preview_chars: int = _PREVIEW_CHARS,
) -> list[TraceHit]:
    return [
        hit_to_trace_hit(hit, rank=index, source=source, preview_chars=preview_chars)
        for index, hit in enumerate(hits, start=1)
    ]


def hit_to_trace_hit(
    hit: KnowledgeChunkHit,
    *,
    rank: int | None = None,
    source: str,
    preview_chars: int = _PREVIEW_CHARS,
) -> TraceHit:
    return TraceHit(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id or None,
        title=hit.title or None,
        text_preview=_preview(hit.text, limit=preview_chars),
        score=float(hit.score) if hit.score is not None else None,
        rank=rank,
        source=source,
        sources=list(hit.sources or hit.lanes or (())),
        metadata={
            "source_path": hit.source_path,
            "heading_path": hit.heading_path,
            "line_start": hit.line_start,
            "line_end": hit.line_end,
            "vector_rank": hit.vector_rank,
            "bm25_rank": hit.bm25_rank,
            "rrf_score": hit.rrf_score,
            "rerank_score": hit.rerank_score,
            "parent_id": hit.parent_id,
            "chunk_type": hit.chunk_type,
            "matched_child_ids": list(hit.matched_child_ids),
        },
    )


def _preview(text: str, *, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"
