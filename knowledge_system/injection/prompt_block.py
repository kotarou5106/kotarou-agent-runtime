from __future__ import annotations

from knowledge_system.indexing.models import KnowledgeChunkHit, KnowledgeCitation


def build_knowledge_prompt_block(
    hits: list[KnowledgeChunkHit],
    citations: list[KnowledgeCitation],
    *,
    max_chars: int = 2400,
) -> tuple[str, list[str]]:
    if not hits or not citations:
        return "", []
    parts = [
        "## Document Knowledge（文档知识）",
        "",
        "以下内容来自本地文档检索结果。回答涉及这些内容时，请优先依据这些片段，并在相关句子后标注引用编号，例如 [K1]。如果片段不足以支持答案，请说明信息不足，不要编造。",
    ]
    injected: list[str] = []
    total = sum(len(part) for part in parts)
    for hit, citation in zip(hits, citations):
        location = _format_location(citation)
        text = _trim_line(hit.text.strip(), 900)
        entry = f"\n[{citation.id}] {location}\n{text}"
        if total + len(entry) > max(400, int(max_chars)):
            break
        parts.append(entry)
        total += len(entry)
        injected.append(hit.chunk_id)
    if not injected:
        return "", []
    return "\n".join(parts).strip(), injected


def _format_location(citation: KnowledgeCitation) -> str:
    heading = f"#{citation.heading_path}" if citation.heading_path else ""
    line_range = ""
    if citation.line_start and citation.line_end:
        line_range = f" 行 {citation.line_start}-{citation.line_end}"
    return f"{citation.source_path}{heading}{line_range}".strip()


def _trim_line(text: str, limit: int) -> str:
    compact = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"
