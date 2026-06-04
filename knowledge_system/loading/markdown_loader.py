from __future__ import annotations

import hashlib
import re
from pathlib import Path

from knowledge_system.indexing.models import LoadedDocument


def load_markdown_document(path: Path) -> LoadedDocument:
    content = path.read_text(encoding="utf-8")
    title = _extract_title(content) or path.stem
    stat = path.stat()
    return LoadedDocument(
        source_path=path,
        title=title,
        content=content,
        file_type="markdown",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        mtime=float(stat.st_mtime),
    )


def _extract_title(content: str) -> str:
    for line in content.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return ""
