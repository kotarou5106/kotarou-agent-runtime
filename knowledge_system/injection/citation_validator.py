from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowledge_system.indexing.models import KnowledgeCitation

_CITATION_RE = re.compile(r"\[K(\d+)\]")


@dataclass(frozen=True)
class CitationValidationResult:
    used_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_ids


def validate_answer_citations(
    answer: str,
    citations: list[KnowledgeCitation] | list[dict[str, object]],
) -> CitationValidationResult:
    used_ids = _dedupe(f"K{match.group(1)}" for match in _CITATION_RE.finditer(answer or ""))
    allowed = {_citation_id(item) for item in citations}
    allowed.discard("")
    valid_ids = [item for item in used_ids if item in allowed]
    missing_ids = [item for item in used_ids if item not in allowed]
    warnings = [
        f"citation {item} was used in the answer but was not present in this turn's retrieved knowledge citations"
        for item in missing_ids
    ]
    return CitationValidationResult(
        used_ids=used_ids,
        missing_ids=missing_ids,
        valid_ids=valid_ids,
        warnings=warnings,
    )


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _citation_id(item: KnowledgeCitation | dict[str, object]) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or "")
    return str(getattr(item, "id", "") or "")
