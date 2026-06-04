from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_system.indexing.models import KnowledgeChunkHit, KnowledgeCitation


@dataclass(frozen=True)
class KnowledgeRetrievalResult:
    block: str = ""
    hits: list[KnowledgeChunkHit] = field(default_factory=list)
    citations: list[KnowledgeCitation] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
