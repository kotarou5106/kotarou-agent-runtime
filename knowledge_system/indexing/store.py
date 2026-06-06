from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_system.indexing.models import DocumentChunk, KnowledgeChunkHit, LoadedDocument


class KnowledgeStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._sqlite_vec: Any | None = None
        self._vec_enabled = False
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema()
            self._try_enable_sqlite_vec()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _ensure_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              source_path TEXT NOT NULL,
              title TEXT,
              content_hash TEXT NOT NULL,
              mtime REAL,
              file_type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active'
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_path
            ON documents(source_path);

            CREATE TABLE IF NOT EXISTS document_chunks (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              heading_path TEXT,
              line_start INTEGER,
              line_end INTEGER,
              token_count INTEGER,
              content_hash TEXT NOT NULL,
              parent_id TEXT,
              chunk_type TEXT NOT NULL DEFAULT 'child',
              metadata_json TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
            ON document_chunks(document_id);

            CREATE TABLE IF NOT EXISTS document_vectors (
              chunk_id TEXT PRIMARY KEY,
              embedding_json TEXT NOT NULL,
              embedding_model TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(chunk_id) REFERENCES document_chunks(id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_retrieval_events (
              id TEXT PRIMARY KEY,
              trace_id TEXT,
              query TEXT NOT NULL,
              retrieved_chunk_ids TEXT NOT NULL,
              injected_chunk_ids TEXT NOT NULL,
              trace_json TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_vector_rows (
              rowid INTEGER PRIMARY KEY AUTOINCREMENT,
              chunk_id TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS knowledge_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        self._ensure_chunk_parent_child_columns()
        self._ensure_retrieval_event_trace_column()
        self._db.commit()

    def _ensure_chunk_parent_child_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self._db.execute("PRAGMA table_info(document_chunks)")
        }
        if "parent_id" not in columns:
            self._db.execute("ALTER TABLE document_chunks ADD COLUMN parent_id TEXT")
        if "chunk_type" not in columns:
            self._db.execute(
                "ALTER TABLE document_chunks ADD COLUMN chunk_type TEXT NOT NULL DEFAULT 'child'"
            )
        if "metadata_json" not in columns:
            self._db.execute("ALTER TABLE document_chunks ADD COLUMN metadata_json TEXT")

    def _ensure_retrieval_event_trace_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self._db.execute("PRAGMA table_info(knowledge_retrieval_events)")
        }
        if "trace_json" not in columns:
            self._db.execute(
                "ALTER TABLE knowledge_retrieval_events ADD COLUMN trace_json TEXT"
            )

    @property
    def vector_backend(self) -> str:
        return "sqlite-vec" if self._vec_enabled else "json-cosine"

    def _try_enable_sqlite_vec(self) -> None:
        try:
            import sqlite_vec  # type: ignore[import-not-found]

            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
            self._sqlite_vec = sqlite_vec
            self._vec_enabled = self._vec_table_exists()
        except Exception:
            try:
                self._db.enable_load_extension(False)
            except Exception:
                pass
            self._sqlite_vec = None
            self._vec_enabled = False

    def _vec_table_exists(self) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_vec_index'"
        ).fetchone()
        return row is not None

    def _ensure_vec_index(self, dim: int) -> bool:
        if self._sqlite_vec is None or dim <= 0:
            return False
        existing_dim = self._get_meta("vector_dim")
        if existing_dim:
            if existing_dim != str(dim):
                self._vec_enabled = False
                return False
        if not self._vec_table_exists():
            try:
                self._db.execute(
                    f"CREATE VIRTUAL TABLE knowledge_vec_index USING vec0(embedding float[{int(dim)}])"
                )
                self._set_meta("vector_dim", str(dim))
                self._db.commit()
            except Exception:
                self._vec_enabled = False
                return False
        self._vec_enabled = True
        return True

    def _get_meta(self, key: str) -> str:
        row = self._db.execute(
            "SELECT value FROM knowledge_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row else ""

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            """
            INSERT INTO knowledge_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def upsert_document_with_chunks(
        self,
        *,
        document_id: str,
        document: LoadedDocument,
        chunks: list[DocumentChunk],
        vectors: dict[str, list[float]] | None = None,
        embedding_model: str = "",
    ) -> None:
        now = _utcnow()
        source_path = str(document.source_path.expanduser().resolve())
        vectors = vectors or {}
        with self._lock:
            existing = self._db.execute(
                "SELECT created_at FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            self._db.execute(
                """
                INSERT INTO documents (
                  id, source_path, title, content_hash, mtime, file_type,
                  created_at, updated_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(id) DO UPDATE SET
                  source_path = excluded.source_path,
                  title = excluded.title,
                  content_hash = excluded.content_hash,
                  mtime = excluded.mtime,
                  file_type = excluded.file_type,
                  updated_at = excluded.updated_at,
                  status = 'active'
                """,
                (
                    document_id,
                    source_path,
                    document.title,
                    document.content_hash,
                    document.mtime,
                    document.file_type,
                    created_at,
                    now,
                ),
            )
            self._db.execute(
                "DELETE FROM document_vectors WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id = ?)",
                (document_id,),
            )
            self._delete_vec_rows_for_document(document_id)
            self._db.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            for chunk in chunks:
                self._db.execute(
                    """
                    INSERT INTO document_chunks (
                      id, document_id, chunk_index, text, heading_path,
                      line_start, line_end, token_count, content_hash,
                      parent_id, chunk_type, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.heading_path,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.token_count,
                        chunk.content_hash,
                        chunk.parent_id,
                        chunk.chunk_type,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        now,
                    ),
                )
                vector = vectors.get(chunk.id)
                if vector:
                    self._db.execute(
                        """
                        INSERT INTO document_vectors (
                          chunk_id, embedding_json, embedding_model, created_at
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            chunk.id,
                            json.dumps(vector),
                            embedding_model,
                            now,
                        ),
                    )
                    self._upsert_vec_row(chunk.id, vector)
            self._db.commit()

    def _delete_vec_rows_for_document(self, document_id: str) -> None:
        rows = self._db.execute(
            """
            SELECT r.rowid
            FROM knowledge_vector_rows r
            JOIN document_chunks c ON c.id = r.chunk_id
            WHERE c.document_id = ?
            """,
            (document_id,),
        ).fetchall()
        if self._vec_enabled:
            for row in rows:
                try:
                    self._db.execute(
                        "DELETE FROM knowledge_vec_index WHERE rowid = ?",
                        (int(row["rowid"]),),
                    )
                except Exception:
                    self._vec_enabled = False
                    break
        self._db.execute(
            """
            DELETE FROM knowledge_vector_rows
            WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)
            """,
            (document_id,),
        )

    def _upsert_vec_row(self, chunk_id: str, vector: list[float]) -> None:
        if not self._ensure_vec_index(len(vector)):
            return
        self._db.execute(
            "INSERT OR IGNORE INTO knowledge_vector_rows(chunk_id) VALUES (?)",
            (chunk_id,),
        )
        row = self._db.execute(
            "SELECT rowid FROM knowledge_vector_rows WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return
        rowid = int(row["rowid"])
        try:
            self._db.execute(
                "DELETE FROM knowledge_vec_index WHERE rowid = ?",
                (rowid,),
            )
            self._db.execute(
                "INSERT INTO knowledge_vec_index(rowid, embedding) VALUES (?, ?)",
                (rowid, self._sqlite_vec.serialize_float32(vector)),
            )
        except Exception:
            self._vec_enabled = False

    def get_document_by_source_path(self, path: str | Path) -> dict[str, Any] | None:
        source_path = str(Path(path).expanduser().resolve())
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, *, status: str = "active") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                WHERE (? = '' OR d.status = ?)
                GROUP BY d.id
                ORDER BY d.updated_at DESC, d.source_path ASC
                """,
                (status, status),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_chunks(self, *, document_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT c.*, d.source_path, d.title
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE (? = '' OR c.document_id = ?)
                ORDER BY d.source_path ASC, c.chunk_index ASC
                LIMIT ?
                """,
                (document_id, document_id, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> KnowledgeChunkHit | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT c.*, d.source_path, d.title
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id = ? AND d.status = 'active'
                """,
                (chunk_id,),
            ).fetchone()
        return _row_to_hit(row, score=0.0, lanes=()) if row else None

    def get_parent_for_child(self, child_id: str) -> KnowledgeChunkHit | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT p.*, d.source_path, d.title
                FROM document_chunks c
                JOIN document_chunks p ON p.id = c.parent_id
                JOIN documents d ON d.id = p.document_id
                WHERE c.id = ? AND d.status = 'active'
                """,
                (child_id,),
            ).fetchone()
        return _row_to_hit(row, score=0.0, lanes=()) if row else None

    def vector_search(
        self,
        *,
        query_vec: list[float],
        top_k: int = 6,
        score_threshold: float = 0.20,
    ) -> list[KnowledgeChunkHit]:
        if self._vec_enabled and self._sqlite_vec is not None:
            try:
                hits = self._vector_search_sqlite_vec(
                    query_vec=query_vec,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                if hits:
                    return hits
            except Exception:
                self._vec_enabled = False
        with self._lock:
            rows = self._db.execute(
                """
                SELECT c.*, d.source_path, d.title, v.embedding_json
                FROM document_vectors v
                JOIN document_chunks c ON c.id = v.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'active'
                  AND COALESCE(c.chunk_type, 'child') != 'parent'
                """
            ).fetchall()
        hits: list[KnowledgeChunkHit] = []
        for row in rows:
            try:
                vector = json.loads(str(row["embedding_json"] or "[]"))
            except json.JSONDecodeError:
                continue
            score = _cosine(query_vec, vector)
            if score < score_threshold:
                continue
            hits.append(_row_to_hit(row, score=score, lanes=("vector",)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]

    def _vector_search_sqlite_vec(
        self,
        *,
        query_vec: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[KnowledgeChunkHit]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT c.*, d.source_path, d.title, v.distance
                FROM knowledge_vec_index v
                JOIN knowledge_vector_rows r ON r.rowid = v.rowid
                JOIN document_chunks c ON c.id = r.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE v.embedding MATCH ? AND k = ? AND d.status = 'active'
                  AND COALESCE(c.chunk_type, 'child') != 'parent'
                ORDER BY v.distance
                """,
                (self._sqlite_vec.serialize_float32(query_vec), max(1, int(top_k))),
            ).fetchall()
        hits: list[KnowledgeChunkHit] = []
        for row in rows:
            distance = float(row["distance"] or 0.0)
            score = 1.0 / (1.0 + max(0.0, distance))
            if score < score_threshold:
                continue
            hits.append(_row_to_hit(row, score=score, lanes=("vector", "sqlite-vec")))
        return hits[: max(1, int(top_k))]

    def keyword_search(self, query: str, *, top_k: int = 12) -> list[KnowledgeChunkHit]:
        from knowledge_system.retrieval.bm25 import bm25_search

        with self._lock:
            rows = self._db.execute(
                """
                SELECT c.*, d.source_path, d.title
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'active'
                  AND COALESCE(c.chunk_type, 'child') != 'parent'
                """
            ).fetchall()
        return bm25_search([dict(row) for row in rows], query, top_k=max(1, int(top_k)))

    def log_retrieval_event(
        self,
        *,
        event_id: str,
        trace_id: str = "",
        query: str,
        retrieved_chunk_ids: list[str],
        injected_chunk_ids: list[str],
        trace: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO knowledge_retrieval_events (
                  id, trace_id, query, retrieved_chunk_ids, injected_chunk_ids,
                  trace_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    trace_id,
                    query,
                    json.dumps(retrieved_chunk_ids),
                    json.dumps(injected_chunk_ids),
                    json.dumps(trace or {}, ensure_ascii=False),
                    _utcnow(),
                ),
            )
            self._db.commit()

    def list_retrieval_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT *
                FROM knowledge_retrieval_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("retrieved_chunk_ids", "injected_chunk_ids"):
                try:
                    item[key] = json.loads(str(item.get(key) or "[]"))
                except json.JSONDecodeError:
                    item[key] = []
            try:
                item["trace"] = json.loads(str(item.get("trace_json") or "{}"))
            except json.JSONDecodeError:
                item["trace"] = {}
            item.pop("trace_json", None)
            items.append(item)
        return items


def _row_to_hit(row: sqlite3.Row, *, score: float, lanes: tuple[str, ...]) -> KnowledgeChunkHit:
    metadata: dict[str, object] = {}
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except Exception:
        metadata = {}
    return KnowledgeChunkHit(
        chunk_id=str(row["id"]),
        document_id=str(row["document_id"]),
        text=str(row["text"] or ""),
        score=float(score),
        source_path=str(row["source_path"] or ""),
        title=str(row["title"] or ""),
        heading_path=str(row["heading_path"] or ""),
        line_start=int(row["line_start"] or 0),
        line_end=int(row["line_end"] or 0),
        lanes=lanes,
        parent_id=str(row["parent_id"] or "") or None,
        chunk_type=str(row["chunk_type"] or "child"),
        metadata=metadata,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
