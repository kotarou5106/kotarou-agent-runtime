from __future__ import annotations

import math
import re
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from knowledge_system.indexing.models import KnowledgeChunkHit

_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class _Bm25Document:
    row: dict[str, Any]
    tokens: list[str]
    term_freq: Counter[str]


class BM25Index:
    """Small in-memory BM25Okapi index for knowledge chunks."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._docs = [_build_doc(row) for row in rows]
        self._doc_count = len(self._docs)
        self._avgdl = (
            sum(len(doc.tokens) for doc in self._docs) / self._doc_count
            if self._doc_count
            else 0.0
        )
        self._doc_freq = _document_frequencies(self._docs)

    def search(self, query: str, *, top_k: int = 12) -> list[KnowledgeChunkHit]:
        """Return chunks ranked by BM25 score for the query."""
        query_terms = tokenize(query)
        if not query_terms or not self._docs:
            return []
        unique_terms = list(dict.fromkeys(query_terms))
        scored: list[tuple[float, _Bm25Document]] = []
        for doc in self._docs:
            score = self._score_doc(unique_terms, doc)
            if score > 0.0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _row_to_hit(doc.row, score=score)
            for score, doc in scored[: max(1, int(top_k))]
        ]

    def _score_doc(self, query_terms: list[str], doc: _Bm25Document) -> float:
        if not doc.tokens or self._avgdl <= 0.0:
            return 0.0
        score = 0.0
        doc_len = len(doc.tokens)
        for term in query_terms:
            freq = doc.term_freq.get(term, 0)
            if freq <= 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score


def bm25_search(
    rows: list[dict[str, Any]],
    query: str,
    *,
    top_k: int = 12,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[KnowledgeChunkHit]:
    """Build a temporary BM25 index over chunk rows and search it."""
    return BM25Index(rows, k1=k1, b=b).search(query, top_k=top_k)


def tokenize(text: str) -> list[str]:
    """Tokenize English words and simple Chinese text for sparse retrieval."""
    raw = str(text or "").lower()
    if not raw.strip():
        return []

    tokens: list[str] = []
    for match in _WORD_RE.finditer(raw):
        token = match.group(0).strip("'")
        if len(token) >= 2:
            tokens.append(token)

    for match in _CJK_RE.finditer(raw):
        cjk = match.group(0)
        tokens.extend(cjk)
        if len(cjk) >= 2:
            tokens.extend(cjk[index : index + 2] for index in range(len(cjk) - 1))
        if len(cjk) <= 8:
            tokens.append(cjk)

    return tokens


def _build_doc(row: dict[str, Any]) -> _Bm25Document:
    text = " ".join(
        [
            str(row.get("text") or ""),
            str(row.get("heading_path") or ""),
            str(row.get("source_path") or ""),
            str(row.get("title") or ""),
        ]
    )
    tokens = tokenize(text)
    return _Bm25Document(row=row, tokens=tokens, term_freq=Counter(tokens))


def _document_frequencies(docs: list[_Bm25Document]) -> dict[str, int]:
    freqs: dict[str, int] = {}
    for doc in docs:
        for token in set(doc.tokens):
            freqs[token] = freqs.get(token, 0) + 1
    return freqs


def _row_to_hit(row: dict[str, Any], *, score: float) -> KnowledgeChunkHit:
    metadata: dict[str, object] = {}
    try:
        metadata = json.loads(str(row.get("metadata_json") or "{}"))
    except Exception:
        metadata = {}
    return KnowledgeChunkHit(
        chunk_id=str(row.get("id") or ""),
        document_id=str(row.get("document_id") or ""),
        text=str(row.get("text") or ""),
        score=float(score),
        source_path=str(row.get("source_path") or ""),
        title=str(row.get("title") or ""),
        heading_path=str(row.get("heading_path") or ""),
        line_start=int(row.get("line_start") or 0),
        line_end=int(row.get("line_end") or 0),
        lanes=("bm25",),
        sources=("bm25",),
        parent_id=str(row.get("parent_id") or "") or None,
        chunk_type=str(row.get("chunk_type") or "child"),
        metadata=metadata,
    )
