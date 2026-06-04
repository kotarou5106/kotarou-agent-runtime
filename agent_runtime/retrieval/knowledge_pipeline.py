from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from knowledge_system.retrieval import KnowledgeQuery


@dataclass(frozen=True)
class KnowledgeRetrievalRequest:
    message: str
    session_key: str
    channel: str
    chat_id: str
    history: list[Any] = field(default_factory=list)
    session_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    trace_id: str = ""


@dataclass(frozen=True)
class KnowledgeRetrievalResult:
    block: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class KnowledgeRetrievalPipeline(Protocol):
    async def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeRetrievalResult: ...


class EmptyKnowledgeRetrievalPipeline:
    async def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeRetrievalResult:
        return KnowledgeRetrievalResult()


class DefaultKnowledgeRetrievalPipeline:
    def __init__(self, service: Any, *, top_k: int = 6) -> None:
        self._service = service
        self._top_k = max(1, int(top_k))

    async def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeRetrievalResult:
        if self._service is None:
            return KnowledgeRetrievalResult()
        try:
            result = await self._service.retrieve(
                KnowledgeQuery(
                    text=request.message,
                    top_k=self._top_k,
                    trace_id=request.trace_id,
                    context={
                        "session_key": request.session_key,
                        "channel": request.channel,
                        "chat_id": request.chat_id,
                        "history": request.history,
                        "session_metadata": request.session_metadata,
                    },
                )
            )
        except Exception:
            return KnowledgeRetrievalResult()
        return KnowledgeRetrievalResult(
            block=result.block,
            citations=[
                {
                    "id": item.id,
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "source_path": item.source_path,
                    "title": item.title,
                    "heading_path": item.heading_path,
                    "line_start": item.line_start,
                    "line_end": item.line_end,
                }
                for item in result.citations
            ],
            trace=dict(result.trace),
        )
