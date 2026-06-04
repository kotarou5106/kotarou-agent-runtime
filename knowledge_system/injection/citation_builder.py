from __future__ import annotations

from knowledge_system.indexing.models import KnowledgeChunkHit, KnowledgeCitation


def build_citations(hits: list[KnowledgeChunkHit]) -> list[KnowledgeCitation]:
    citations: list[KnowledgeCitation] = []
    for index, hit in enumerate(hits, start=1):
        citations.append(
            KnowledgeCitation(
                id=f"K{index}",
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_path=hit.source_path,
                title=hit.title,
                heading_path=hit.heading_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
            )
        )
    return citations
