from __future__ import annotations

from typing import Any

from knowledge_system.indexing.models import KnowledgeCitation
from knowledge_system.injection.citation_validator import CitationValidationResult
from knowledge_system.retrieval.trace import CitationValidationTrace


def citation_blocks_to_trace(citations: list[KnowledgeCitation]) -> list[dict[str, Any]]:
    """Format citations for trace storage without duplicating chunk text."""
    return [
        {
            "id": citation.id,
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "title": citation.title,
            "source_path": citation.source_path,
            "heading_path": citation.heading_path,
            "line_start": citation.line_start,
            "line_end": citation.line_end,
            "parent_id": citation.parent_id,
            "matched_child_ids": list(citation.matched_child_ids),
        }
        for citation in citations
    ]


def citation_validation_to_trace(
    result: CitationValidationResult,
    citations: list[KnowledgeCitation],
) -> CitationValidationTrace:
    allowed_ids = [citation.id for citation in citations]
    unused = [citation_id for citation_id in allowed_ids if citation_id not in result.used_ids]
    message = "; ".join(result.warnings) if result.warnings else None
    return CitationValidationTrace(
        is_valid=result.ok,
        missing_citations=list(result.missing_ids),
        unused_citations=unused,
        invalid_citations=list(result.missing_ids),
        message=message,
    )
