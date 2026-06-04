from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeQuery:
    text: str
    top_k: int = 6
    trace_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
