from __future__ import annotations

from dataclasses import replace

from knowledge_system.indexing.models import KnowledgeChunkHit
from knowledge_system.indexing.store import KnowledgeStore


def expand_hits_to_parents(
    hits: list[KnowledgeChunkHit],
    *,
    store: KnowledgeStore,
    enabled: bool = True,
) -> tuple[list[KnowledgeChunkHit], dict[str, object]]:
    """Map child hits to parent chunks for context injection."""
    trace: dict[str, object] = {
        "enabled": bool(enabled),
        "fallback_count": 0,
        "child_to_parent": {},
        "matched_child_ids": {},
    }
    if not enabled or not hits:
        trace["skip_reason"] = "disabled" if not enabled else "no hits"
        return hits, trace

    by_parent: dict[str, KnowledgeChunkHit] = {}
    matched_children: dict[str, list[str]] = {}
    child_to_parent: dict[str, str] = {}
    fallback_count = 0

    for hit in hits:
        parent = None
        if hit.parent_id:
            parent = store.get_chunk(hit.parent_id)
        elif hit.chunk_type == "child":
            parent = store.get_parent_for_child(hit.chunk_id)
        if parent is None:
            fallback_count += 1
            parent_key = hit.chunk_id
            parent_hit = _with_child_match(hit, hit.chunk_id, score=hit.score)
        else:
            parent_key = parent.chunk_id
            child_to_parent[hit.chunk_id] = parent.chunk_id
            parent_hit = _with_child_match(parent, hit.chunk_id, score=hit.score, child_hit=hit)

        existing = by_parent.get(parent_key)
        if existing is None or parent_hit.score > existing.score:
            by_parent[parent_key] = parent_hit
        matched_children.setdefault(parent_key, [])
        for child_id in parent_hit.matched_child_ids:
            if child_id not in matched_children[parent_key]:
                matched_children[parent_key].append(child_id)

    expanded: list[KnowledgeChunkHit] = []
    for parent_key, hit in by_parent.items():
        children = tuple(matched_children.get(parent_key, list(hit.matched_child_ids)))
        metadata = dict(hit.metadata)
        metadata["matched_child_ids"] = list(children)
        expanded.append(
            replace(
                hit,
                matched_child_ids=children,
                metadata=metadata,
                chunk_type="parent" if parent_key != hit.chunk_id or hit.chunk_type == "parent" else hit.chunk_type,
            )
        )
    expanded.sort(key=lambda item: item.score, reverse=True)
    trace["fallback_count"] = fallback_count
    trace["child_to_parent"] = child_to_parent
    trace["matched_child_ids"] = matched_children
    return expanded, trace


def _with_child_match(
    hit: KnowledgeChunkHit,
    child_id: str,
    *,
    score: float,
    child_hit: KnowledgeChunkHit | None = None,
) -> KnowledgeChunkHit:
    matched = tuple(dict.fromkeys((*hit.matched_child_ids, child_id)))
    metadata = dict(hit.metadata)
    metadata["matched_child_ids"] = list(matched)
    if child_hit is not None:
        metadata["child_scores"] = {
            **dict(metadata.get("child_scores") or {}),
            child_id: child_hit.score,
        }
        metadata["child_sources"] = {
            **dict(metadata.get("child_sources") or {}),
            child_id: list(child_hit.sources or child_hit.lanes),
        }
    return replace(
        hit,
        score=max(float(hit.score), float(score)),
        matched_child_ids=matched,
        metadata=metadata,
    )
