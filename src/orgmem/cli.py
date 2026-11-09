"""Command-line interface: ``orgmem <command>``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .models import Status
from .store import Store

DEFAULT_DB = Path(os.environ.get("ORGMEM_DB", ".orgmem/memory.db"))


def _print_memory(mem, verbose: bool = False) -> None:
    flag = {Status.CONFLICTED: " [CONFLICTED]", Status.SUPERSEDED: " [SUPERSEDED]"}
    print(f"{mem.id}  {mem.kind.value:<9}  {mem.subject}{flag.get(mem.status, '')}")
    print(f"    {mem.statement}")
    for s in mem.sources if verbose else mem.sources[:1]:
        print(f"    <- {s.doc_path}:{s.locator} ({s.doc_date})")
        if verbose:
            print(f'       "{s.quote}"')
    if not verbose and len(mem.sources) > 1:
        print(f"    (+{len(mem.sources) - 1} more sources)")


def cmd_ingest(args, store: Store) -> int:
    from .llm import AnthropicLLM
    from .pipeline import ingest_path

    llm = AnthropicLLM(model=args.model)
    report = ingest_path(
        store, llm, llm, Path(args.path), on_progress=lambda p: print(f"ingesting {p}")
    )
    print(
        f"\n{report.documents} documents ingested ({report.skipped_unchanged} unchanged, skipped)\n"
        f"{report.extracted} claims extracted, {report.ungrounded} dropped as ungrounded\n"
        f"{report.added} added, {report.duplicates} merged as duplicates, "
        f"{report.updated} updated older memories, {report.conflicts} conflicts flagged"
    )
    for err in report.errors:
        print(f"error: {err}", file=sys.stderr)
    return 1 if report.errors else 0


def cmd_search(args, store: Store) -> int:
    results = store.search(args.query, limit=args.limit)
    if not results:
        print("No matching memories.")
    for mem in results:
        _print_memory(mem)
    return 0


def cmd_show(args, store: Store) -> int:
    mem = store.get(args.id)
    if mem is None:
        print(f"No memory {args.id}", file=sys.stderr)
        return 1
    _print_memory(mem, verbose=True)
    history = store.history(mem.id)
    if history:
        print("\n  replaced:")
        for old in history:
            print(f"    {old.id}  {old.statement}  ({old.latest_date})")
    return 0


def cmd_conflicts(args, store: Store) -> int:
    conflicts = store.open_conflicts()
    if not conflicts:
        print("No open conflicts.")
    for c in conflicts:
        print(f"{c.id}: {c.reason}")
        for mid in (c.memory_a, c.memory_b):
            _print_memory(store.get(mid))
        print(f"  resolve with: orgmem resolve {c.id} --keep <memory id>\n")
    return 0


def cmd_resolve(args, store: Store) -> int:
    try:
        store.resolve_conflict(args.conflict_id, args.keep)
    except (KeyError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Kept {args.keep}; the other memory is now superseded.")
    return 0


def cmd_export(args, store: Store) -> int:
    from .export import render_markdown

    text = render_markdown(store)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


def cmd_stats(args, store: Store) -> int:
    for key, value in store.stats().items():
        print(f"{key:<16}{value}")
    return 0


def cmd_serve(args, store: Store) -> int:
    try:
        from .mcp_server import serve
    except ImportError:
        print("The MCP server needs the extra: pip install 'orgmem[mcp]'", file=sys.stderr)
        return 1
    store.close()  # the server opens its own connection
    serve(args.db)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orgmem", description=__doc__)
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"database file (default {DEFAULT_DB})"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="log pipeline decisions")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="extract and consolidate memories from files")
    s.add_argument("path", help="a file or a directory of .md/.txt/.json files")
    s.add_argument("--model", help="Claude model id (default: $ORGMEM_MODEL or claude-opus-5-5)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="search the memory")
    s.add_argument("query")
    s.add_argument("-n", "--limit", type=int, default=10)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("show", help="show one memory with all sources and history")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("conflicts", help="list contradictions waiting for review")
    s.set_defaults(func=cmd_conflicts)

    s = sub.add_parser("resolve", help="resolve a conflict by keeping one memory")
    s.add_argument("conflict_id")
    s.add_argument("--keep", required=True, help="id of the memory that is correct")
    s.set_defaults(func=cmd_resolve)

    s = sub.add_parser("export", help="write the memory as Markdown")
    s.add_argument("-o", "--output", default="ORG_MEMORY.md", help="file path, or - for stdout")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("stats", help="counts of memories, sources, documents, conflicts")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("serve", help="run the MCP server (stdio) for AI assistants")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    store = Store(args.db)
    try:
        return args.func(args, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
