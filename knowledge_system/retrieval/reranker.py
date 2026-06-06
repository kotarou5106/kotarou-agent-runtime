from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from typing import Any, Protocol

from knowledge_system.indexing.models import KnowledgeChunkHit


class BaseReranker(Protocol):
    async def rerank(
        self,
        query: str,
        hits: list[KnowledgeChunkHit],
        *,
        top_k: int,
    ) -> list[KnowledgeChunkHit]: ...


class NoOpReranker:
    async def rerank(
        self,
        query: str,
        hits: list[KnowledgeChunkHit],
        *,
        top_k: int,
    ) -> list[KnowledgeChunkHit]:
        return hits[: max(1, int(top_k))]


class LLMReranker:
    """Lightweight LLM reranker that scores candidates without generating answers."""

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        top_n: int = 10,
        timeout_seconds: int = 15,
        preview_chars: int = 700,
    ) -> None:
        self._provider = provider
        self._model = model
        self._top_n = max(1, int(top_n))
        self._timeout_seconds = max(1, int(timeout_seconds))
        self._preview_chars = max(120, int(preview_chars))
        self.last_warning = ""

    async def rerank(
        self,
        query: str,
        hits: list[KnowledgeChunkHit],
        *,
        top_k: int,
    ) -> list[KnowledgeChunkHit]:
        self.last_warning = ""
        if not hits:
            return []
        candidates = hits[: self._top_n]
        try:
            scores = await asyncio.wait_for(
                self._score(query, candidates),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            self.last_warning = f"llm reranker failed; using original order: {exc}"
            return hits[: max(1, int(top_k))]
        if not scores:
            self.last_warning = "llm reranker returned no scores; using original order"
            return hits[: max(1, int(top_k))]

        reranked: list[KnowledgeChunkHit] = []
        for hit in candidates:
            score = scores.get(hit.chunk_id)
            if score is None:
                score = 0.0
            metadata = dict(hit.metadata)
            metadata["rerank_score"] = score
            reranked.append(replace(hit, rerank_score=score, metadata=metadata))
        reranked.sort(key=lambda item: (item.rerank_score or 0.0, item.score), reverse=True)
        result = [*reranked, *hits[len(candidates):]]
        return result[: max(1, int(top_k))] or hits[: max(1, int(top_k))]

    async def _score(
        self,
        query: str,
        hits: list[KnowledgeChunkHit],
    ) -> dict[str, float]:
        response = await self._provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict retrieval reranker. Score each candidate only "
                        "by relevance to the query. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_prompt(query, hits, preview_chars=self._preview_chars),
                },
            ],
            tools=[],
            model=self._model,
            max_tokens=400,
            tool_choice="none",
            disable_thinking=True,
        )
        content = str(getattr(response, "content", "") or "")
        payload = _parse_json_object(content)
        items = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return {}
        scores: dict[str, float] = {}
        allowed = {hit.chunk_id for hit in hits}
        for item in items:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("id") or "")
            if chunk_id not in allowed:
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            scores[chunk_id] = min(1.0, max(0.0, score))
        return scores


def _build_prompt(query: str, hits: list[KnowledgeChunkHit], *, preview_chars: int) -> str:
    candidates = [
        {
            "id": hit.chunk_id,
            "text": _preview(hit.text, preview_chars),
        }
        for hit in hits
    ]
    return json.dumps(
        {
            "task": "Score each candidate from 0 to 1 for relevance to the query. If unsure, use a low score. Return only {'scores': [{'id': str, 'score': number}]} JSON.",
            "query": query,
            "candidates": candidates,
        },
        ensure_ascii=False,
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _preview(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"
