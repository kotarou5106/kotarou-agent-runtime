from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")

_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("工具", "tool tool calling function calling"),
    ("tool", "tool calling function calling"),
    ("记忆", "memory long-term memory retrieval"),
    ("memory", "memory long-term memory retrieval"),
    ("检索", "retrieval vector search bm25 hybrid search"),
    ("retrieval", "retrieval vector search bm25 hybrid search"),
    ("引用", "citation source validation"),
    ("citation", "citation source validation"),
    ("任务", "task scheduler background task proactive notification"),
    ("scheduler", "task scheduler background task proactive notification"),
    ("插件", "plugin plugin system lifecycle hook"),
    ("plugin", "plugin plugin system lifecycle hook"),
)


def rewrite_query(query: str, *, max_variants: int = 3) -> list[str]:
    """Return conservative retrieval-oriented variants without changing intent."""
    original = str(query or "").strip()
    if not original:
        return []

    normalized = _normalize(original)
    expansions = _matching_expansions(normalized)
    variants = [original]
    if normalized and normalized != original:
        variants.append(normalized)
    if expansions:
        variants.append(_normalize(f"{normalized} {' '.join(expansions)}"))

    return _dedupe(variants)[: max(1, int(max_variants))]


def _normalize(query: str) -> str:
    return _SPACE_RE.sub(" ", query.strip().lower())


def _matching_expansions(normalized_query: str) -> list[str]:
    expansions: list[str] = []
    for trigger, expansion in _EXPANSIONS:
        if trigger in normalized_query:
            expansions.append(expansion)
    return _dedupe(expansions)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = item.strip()
        key = _normalize(value)
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
