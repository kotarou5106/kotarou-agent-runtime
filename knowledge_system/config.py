from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeConfig:
    enabled: bool = False
    db_path: str = "knowledge.db"
    top_k: int = 6
    score_threshold: float = 0.20
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 120
    inject_max_chars: int = 2400


def resolve_knowledge_db_path(workspace: Path, db_path: str = "knowledge.db") -> Path:
    path = Path(db_path)
    if path.is_absolute():
        return path
    return workspace / path
