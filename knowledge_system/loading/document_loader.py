from __future__ import annotations

from pathlib import Path

from knowledge_system.indexing.models import LoadedDocument
from knowledge_system.loading.markdown_loader import load_markdown_document
from knowledge_system.loading.text_loader import load_text_document


class DocumentLoader:
    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

    def load(self, path: str | Path) -> LoadedDocument:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        suffix = source.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return load_markdown_document(source)
        if suffix == ".txt":
            return load_text_document(source)
        raise ValueError(f"unsupported document type: {suffix}")
