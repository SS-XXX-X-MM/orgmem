"""SQLite storage with full-text search (FTS5).

One file on disk, no server, nothing to deploy. FTS5 gives ranked keyword
search, which we use both to answer queries and to find existing memories a new
claim might duplicate or contradict.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import Conflict, Kind, Memory, Source, Status

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    subject       TEXT NOT NULL,
    statement     TEXT NOT NULL,
    status        TEXT NOT NULL,
    superseded_by TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    memory_id TEXT NOT NULL REFERENCES memories(id),
    doc_path  TEXT NOT NULL,
    locator   TEXT NOT NULL,
    quote     TEXT NOT NULL,
    doc_date  TEXT NOT NULL,
    UNIQUE (memory_id, doc_path, quote)
);
CREATE TABLE IF NOT EXISTS documents (
    path        TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflicts (
    id       TEXT PRIMARY KEY,
    memory_a TEXT NOT NULL REFERENCES memories(id),
    memory_b TEXT NOT NULL REFERENCES memories(id),
    reason   TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED, subject, statement
);
"""

_WORD = re.compile(r"\w+", re.UNICODE)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: every word quoted, OR-ed together.

    Quoting stops user input like ``AND`` or ``"`` from being read as FTS syntax.
    """
    words = {w.lower() for w in _WORD.findall(text) if len(w) > 2}
    return " OR ".join(f'"{w}"' for w in sorted(words))


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the MCP server may run tool calls on worker threads.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- documents ------------------------------------------------------------

    def is_unchanged(self, path: str, sha256: str) -> bool:
        row = self.db.execute("SELECT sha256 FROM documents WHERE path = ?", (path,)).fetchone()
        return row is not None and row["sha256"] == sha256

    def mark_ingested(self, path: str, sha256: str) -> None:
        self.db.execute(
            "INSERT INTO documents (path, sha256, ingested_at) VALUES (?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET sha256 = excluded.sha256, "
            "ingested_at = excluded.ingested_at",
            (path, sha256, _now()),
        )
        self.db.commit()

    # --- memories -------------------------------------------------------------

    def add_memory(self, kind: Kind, subject: str, statement: str, source: Source) -> Memory:
        now = _now()
        mem = Memory(
            id=_new_id("m"),
            kind=kind,
            subject=subject.strip().lower(),
            statement=statement.strip(),
            status=Status.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        self.db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (mem.id, mem.kind.value, mem.subject, mem.statement, mem.status.value, now, now),
        )
        self.db.execute(
            "INSERT INTO memories_fts (memory_id, subject, statement) VALUES (?, ?, ?)",
            (mem.id, mem.subject, mem.statement),
        )
        self.add_source(mem.id, source)  # commits
        mem.sources = [source]
        return mem

    def add_source(self, memory_id: str, source: Source) -> None:
        """Attach another piece of evidence. A repeat of the same quote is ignored."""
        self.db.execute(
            "INSERT OR IGNORE INTO sources VALUES (?, ?, ?, ?, ?)",
            (memory_id, source.doc_path, source.locator, source.quote, source.doc_date),
        )
        self.db.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (_now(), memory_id))
        self.db.commit()

    def supersede(self, old_id: str, new_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            (Status.SUPERSEDED.value, new_id, _now(), old_id),
        )
        self.db.commit()

    def set_status(self, memory_id: str, status: Status) -> None:
        self.db.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, _now(), memory_id),
        )
        self.db.commit()

    def get(self, memory_id: str) -> Memory | None:
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._hydrate(row) if row else None

    def all(
        self, statuses: tuple[Status, ...] = (Status.ACTIVE, Status.CONFLICTED)
    ) -> list[Memory]:
        marks = ",".join("?" * len(statuses))
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE status IN ({marks}) ORDER BY kind, subject, created_at",
            [s.value for s in statuses],
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    def search(
        self,
        text: str,
        limit: int = 10,
        statuses: tuple[Status, ...] = (Status.ACTIVE, Status.CONFLICTED),
        exclude: str | None = None,
    ) -> list[Memory]:
        """Best-matching memories by BM25 rank (lower bm25() = better match)."""
        query = fts_query(text)
        if not query:
            return []
        marks = ",".join("?" * len(statuses))
        rows = self.db.execute(
            f"""
            SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.memory_id
            WHERE memories_fts MATCH ? AND m.status IN ({marks}) AND m.id != ?
            ORDER BY bm25(memories_fts) LIMIT ?
            """,
            [query, *[s.value for s in statuses], exclude or "", limit],
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    def history(self, memory_id: str) -> list[Memory]:
        """Older memories this one replaced, newest first."""
        chain, current = [], memory_id
        while row := self.db.execute(
            "SELECT * FROM memories WHERE superseded_by = ?", (current,)
        ).fetchone():
            mem = self._hydrate(row)
            chain.append(mem)
            current = mem.id
        return chain

    # --- conflicts ------------------------------------------------------------

    def add_conflict(self, memory_a: str, memory_b: str, reason: str) -> Conflict:
        conflict = Conflict(id=_new_id("c"), memory_a=memory_a, memory_b=memory_b, reason=reason)
        self.db.execute(
            "INSERT INTO conflicts VALUES (?, ?, ?, ?, 0)",
            (conflict.id, memory_a, memory_b, reason),
        )
        for mid in (memory_a, memory_b):
            self.db.execute(
                "UPDATE memories SET status = ? WHERE id = ?", (Status.CONFLICTED.value, mid)
            )
        self.db.commit()
        return conflict

    def open_conflicts(self) -> list[Conflict]:
        rows = self.db.execute("SELECT * FROM conflicts WHERE resolved = 0").fetchall()
        return [
            Conflict(r["id"], r["memory_a"], r["memory_b"], r["reason"], bool(r["resolved"]))
            for r in rows
        ]

    def resolve_conflict(self, conflict_id: str, keep_id: str) -> None:
        row = self.db.execute("SELECT * FROM conflicts WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            raise KeyError(f"No conflict {conflict_id}")
        if keep_id not in (row["memory_a"], row["memory_b"]):
            raise ValueError(f"{keep_id} is not part of conflict {conflict_id}")
        drop_id = row["memory_b"] if keep_id == row["memory_a"] else row["memory_a"]
        self.db.execute("UPDATE conflicts SET resolved = 1 WHERE id = ?", (conflict_id,))
        self.supersede(drop_id, keep_id)  # commits
        # A memory can sit in several conflicts; only clear the flag once none are left open.
        still_open = self.db.execute(
            "SELECT 1 FROM conflicts WHERE resolved = 0 AND (memory_a = ? OR memory_b = ?)",
            (keep_id, keep_id),
        ).fetchone()
        if not still_open:
            self.set_status(keep_id, Status.ACTIVE)

    # --- stats ----------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        q = self.db.execute
        out = {
            row["status"]: row["n"]
            for row in q("SELECT status, COUNT(*) AS n FROM memories GROUP BY status")
        }
        out["documents"] = q("SELECT COUNT(*) FROM documents").fetchone()[0]
        out["sources"] = q("SELECT COUNT(*) FROM sources").fetchone()[0]
        out["open_conflicts"] = q("SELECT COUNT(*) FROM conflicts WHERE resolved = 0").fetchone()[0]
        return out

    # --- helpers --------------------------------------------------------------

    def _hydrate(self, row: sqlite3.Row) -> Memory:
        srcs = self.db.execute(
            "SELECT doc_path, locator, quote, doc_date FROM sources WHERE memory_id = ? "
            "ORDER BY doc_date",
            (row["id"],),
        ).fetchall()
        return Memory(
            id=row["id"],
            kind=Kind(row["kind"]),
            subject=row["subject"],
            statement=row["statement"],
            status=Status(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=row["superseded_by"],
            sources=[Source(**dict(s)) for s in srcs],
        )
