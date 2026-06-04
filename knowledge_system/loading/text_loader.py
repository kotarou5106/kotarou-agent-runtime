from __future__ import annotations

import hashlib
from pathlib import Path

from knowledge_system.indexing.models import LoadedDocument


def load_text_document(path: Path) -> LoadedDocument:
    content = path.read_text(encoding="utf-8")
    stat = path.stat()
    return LoadedDocument(
        source_path=path,
        title=path.stem,
        content=content,
        file_type="text",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        mtime=float(stat.st_mtime),
    )
