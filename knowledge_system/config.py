from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeRerankingConfig:
    enabled: bool = False
    type: str = "none"
    top_n: int = 10
    timeout_seconds: int = 15


@dataclass(frozen=True)
class KnowledgeParentChildConfig:
    enabled: bool = True
    expand_to_parent: bool = True
    parent_chunk_size: int = 1800
    parent_chunk_overlap: int = 180
    child_chunk_size: int = 600
    child_chunk_overlap: int = 80


@dataclass(frozen=True)
class KnowledgeConfig:
    enabled: bool = False
    db_path: str = "knowledge.db"
    top_k: int = 6
    score_threshold: float = 0.20
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 120
    inject_max_chars: int = 2400
    reranking: KnowledgeRerankingConfig = field(default_factory=KnowledgeRerankingConfig)
    parent_child: KnowledgeParentChildConfig = field(default_factory=KnowledgeParentChildConfig)


def resolve_knowledge_db_path(workspace: Path, db_path: str = "knowledge.db") -> Path:
    path = Path(db_path)
    if path.is_absolute():
        return path
    return workspace / path
