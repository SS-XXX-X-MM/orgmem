"""Expose the shared memory to any MCP-capable AI tool (Claude, Cursor, IDEs, agents).

This is what turns orgmem from a file into infrastructure: every assistant in the
company can read the same memory, and ``remember`` lets a conversation feed new
knowledge back in, where it goes through the same grounding and consolidation
as ingested documents.

Run with ``orgmem serve`` (stdio transport).
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import ingest
from .llm import AnthropicLLM
from .models import Memory
from .pipeline import Report, ingest_document
from .store import Store


def _as_dict(mem: Memory) -> dict:
    return {
        "id": mem.id,
        "kind": mem.kind.value,
        "subject": mem.subject,
        "statement": mem.statement,
        "status": mem.status.value,
        "sources": [f"{s.doc_path}:{s.locator} ({s.doc_date})" for s in mem.sources],
    }


def build_server(db_path: Path) -> MCPServer:
    server = MCPServer(
        name="orgmem",
        instructions=(
            "Shared organisational memory. Call search_memory before answering questions "
            "about how this company works, who owns what, or what was decided. Cite the "
            "sources it returns. Treat memories with status 'conflicted' as disputed."
        ),
    )
    store = Store(db_path)
    llm: AnthropicLLM | None = None  # created on first write, so reads never need an API key

    @server.tool()
    def search_memory(query: str, limit: int = 8) -> list[dict]:
        """Search the organisation's shared memory (decisions, processes, owners, facts,
        glossary). Returns the best matches with their sources."""
        return [_as_dict(m) for m in store.search(query, limit=max(1, min(limit, 25)))]

    @server.tool()
    def get_memory(memory_id: str) -> dict:
        """Fetch one memory with its sources and the older memories it replaced."""
        mem = store.get(memory_id)
        if mem is None:
            return {"error": f"No memory {memory_id}"}
        return {**_as_dict(mem), "replaced": [_as_dict(m) for m in store.history(memory_id)]}

    @server.tool()
    def list_conflicts() -> list[dict]:
        """List contradictions between memories that are waiting for a human to resolve."""
        return [
            {
                "id": c.id,
                "reason": c.reason,
                "a": _as_dict(store.get(c.memory_a)),
                "b": _as_dict(store.get(c.memory_b)),
            }
            for c in store.open_conflicts()
        ]

    @server.tool()
    def remember(note: str, source: str) -> dict:
        """Add knowledge learned in a conversation to the shared memory. `note` should
        state durable facts or decisions in full sentences; `source` names where it came
        from (e.g. "chat with Priya about billing")."""
        nonlocal llm
        llm = llm or AnthropicLLM()
        report = Report()
        ingest_document(store, llm, llm, ingest.from_text(note, f"mcp:{source}"), report)
        return {
            "added": report.added,
            "duplicates": report.duplicates,
            "updated": report.updated,
            "conflicts": report.conflicts,
            "dropped_ungrounded": report.ungrounded,
        }

    return server


def serve(db_path: Path) -> None:
    build_server(db_path).run()
