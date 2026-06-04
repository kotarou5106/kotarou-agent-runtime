from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagEvalCase:
    query: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_document_ids: list[str] = field(default_factory=list)
    k: int = 5


@dataclass(frozen=True)
class RagEvalResult:
    recall_at_k: float
    citation_hit: bool
    hit_chunk_ids: list[str]
    hit_document_ids: list[str]
    cited_ids: list[str]


def evaluate_rag_case(
    case: RagEvalCase,
    *,
    retrieved_chunks: list[dict[str, object]],
    answer: str = "",
    citations: list[dict[str, object]] | None = None,
) -> RagEvalResult:
    top = retrieved_chunks[: max(1, int(case.k))]
    retrieved_chunk_ids = [str(item.get("chunk_id") or item.get("id") or "") for item in top]
    retrieved_document_ids = [str(item.get("document_id") or "") for item in top]
    gold_chunks = {item for item in case.gold_chunk_ids if item}
    gold_docs = {item for item in case.gold_document_ids if item}
    chunk_hits = [item for item in retrieved_chunk_ids if item in gold_chunks]
    doc_hits = [item for item in retrieved_document_ids if item in gold_docs]
    target_count = len(gold_chunks) + len(gold_docs)
    hit_count = len(set(chunk_hits)) + len(set(doc_hits))
    recall = 1.0 if target_count == 0 else min(1.0, hit_count / target_count)
    cited_ids = _extract_citation_ids(answer)
    allowed_citation_ids = {
        str(item.get("id") or "")
        for item in (citations or [])
        if str(item.get("id") or "")
    }
    citation_hit = bool(cited_ids) and all(item in allowed_citation_ids for item in cited_ids)
    return RagEvalResult(
        recall_at_k=recall,
        citation_hit=citation_hit,
        hit_chunk_ids=chunk_hits,
        hit_document_ids=doc_hits,
        cited_ids=cited_ids,
    )


def _extract_citation_ids(answer: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r"\[K(\d+)\]", answer or ""):
        value = f"K{match.group(1)}"
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
