"""Load documents from disk and split them into chunks the LLM can read.

Supported inputs:

* ``.md`` / ``.txt``: notes, docs, meeting minutes.
* ``.json``: chat exports shaped like
  ``{"title": "...", "date": "YYYY-MM-DD", "messages": [{"author": "...", "text": "..."}]}``.

Every chunk keeps its line range, so each memory can cite where it came from.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}
MAX_CHUNK_CHARS = 6000

_DATE_IN_TEXT = re.compile(r"^date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.IGNORECASE | re.MULTILINE)
_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class Document:
    path: str
    text: str
    doc_date: str
    sha256: str


@dataclass
class Chunk:
    doc: Document
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int

    @property
    def locator(self) -> str:
        return f"L{self.start_line}-L{self.end_line}"


def discover(root: Path) -> list[Path]:
    """Return supported files under ``root`` (or ``root`` itself), sorted for determinism."""
    if root.is_file():
        return [root] if root.suffix in SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    )


def load(path: Path, display_path: str | None = None) -> Document:
    raw = path.read_text(encoding="utf-8")
    text = _chat_to_text(raw) if path.suffix == ".json" else raw
    return Document(
        path=display_path or str(path),
        text=text,
        doc_date=_detect_date(path, raw),
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def from_text(text: str, source_name: str, doc_date: str | None = None) -> Document:
    """Build a document from free text, e.g. a note an AI assistant asked us to remember."""
    return Document(
        path=source_name,
        text=text,
        doc_date=doc_date or date.today().isoformat(),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def chunk(doc: Document, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split on blank lines, packing paragraphs into chunks of at most ``max_chars``.

    Splitting at paragraph boundaries keeps a decision and its context together;
    a single paragraph longer than ``max_chars`` becomes its own chunk.
    """
    lines = doc.text.splitlines()
    paragraphs: list[tuple[int, int]] = []  # (start, end) 0-based inclusive line indexes
    start = None
    for i, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            paragraphs.append((start, i - 1))
            start = None
    if start is not None:
        paragraphs.append((start, len(lines) - 1))

    chunks: list[Chunk] = []
    cur_start = cur_end = None
    cur_len = 0
    for p_start, p_end in paragraphs:
        p_len = sum(len(lines[j]) + 1 for j in range(p_start, p_end + 1))
        if cur_start is not None and cur_len + p_len > max_chars:
            chunks.append(_make_chunk(doc, lines, cur_start, cur_end))
            cur_start, cur_len = None, 0
        if cur_start is None:
            cur_start = p_start
        cur_end = p_end
        cur_len += p_len
    if cur_start is not None:
        chunks.append(_make_chunk(doc, lines, cur_start, cur_end))
    return chunks


def _make_chunk(doc: Document, lines: list[str], start: int, end: int) -> Chunk:
    return Chunk(
        doc=doc, text="\n".join(lines[start : end + 1]), start_line=start + 1, end_line=end + 1
    )


def _chat_to_text(raw: str) -> str:
    data = json.loads(raw)
    out = []
    if data.get("title"):
        out += [f"# {data['title']}", ""]
    for msg in data.get("messages", []):
        out.append(f"{msg.get('author', 'unknown')}: {msg.get('text', '')}")
        out.append("")  # blank line = paragraph boundary, so long chats chunk cleanly
    return "\n".join(out)


def _detect_date(path: Path, raw: str) -> str:
    """Prefer an explicit date (front matter / JSON field), then the filename, then mtime."""
    if path.suffix == ".json":
        try:
            if d := json.loads(raw).get("date"):
                return str(d)[:10]
        except (json.JSONDecodeError, AttributeError):
            pass
    if m := _DATE_IN_TEXT.search(raw[:2000]):
        return m.group(1)
    if m := _DATE_IN_NAME.search(path.name):
        return m.group(1)
    return date.fromtimestamp(path.stat().st_mtime).isoformat()
