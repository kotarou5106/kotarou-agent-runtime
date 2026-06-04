from __future__ import annotations

from typing import Protocol


class EmbedsText(Protocol):
    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class KnowledgeEmbedderAdapter:
    def __init__(self, embedder: EmbedsText) -> None:
        self._embedder = embedder

    async def embed_text(self, text: str) -> list[float]:
        return await self._embedder.embed(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embed_batch = getattr(self._embedder, "embed_batch", None)
        if callable(embed_batch):
            return await embed_batch(texts)
        return [await self._embedder.embed(text) for text in texts]
