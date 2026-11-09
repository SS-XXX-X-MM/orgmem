"""The ingest pipeline: document -> chunks -> extracted claims -> consolidated memory.

for each chunk:
    claims = extractor.extract(chunk)
    for claim in claims:
        drop it if its evidence quote is not really in the chunk     (grounding)
        find similar existing memories via full-text search          (retrieval)
        near-identical wording? -> duplicate, no LLM call            (shortcut)
        otherwise ask the judge: add / duplicate / update / conflict (consolidation)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import ingest
from .llm import Extractor, Judge
from .models import Decision, ExtractedMemory, Memory, Source, Status
from .store import Store

log = logging.getLogger(__name__)

CANDIDATES = 5
DUPLICATE_JACCARD = 0.85  # word-overlap above this is treated as the same claim


@dataclass
class Report:
    documents: int = 0
    skipped_unchanged: int = 0
    extracted: int = 0
    ungrounded: int = 0
    added: int = 0
    duplicates: int = 0
    updated: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)


# --- grounding ----------------------------------------------------------------


def _normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"')
    text = text.replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def is_grounded(evidence: str, chunk_text: str) -> bool:
    """True if the quoted evidence really appears in the source text.

    LLMs sometimes paraphrase or invent "quotes". Checking the quote against the
    source is cheap, needs no extra model call, and keeps every stored memory
    traceable to real text. Whitespace, case and curly quotes are ignored.
    """
    quote = _normalise(evidence).strip("\"' .")
    return len(quote) >= 8 and quote in _normalise(chunk_text)


# --- consolidation ------------------------------------------------------------


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def consolidate(
    store: Store, judge: Judge, claim: ExtractedMemory, source: Source, report: Report
) -> Memory:
    """Merge one grounded claim into the store and return the memory it ended up in."""
    candidates = store.search(f"{claim.subject} {claim.statement}", limit=CANDIDATES)

    for cand in candidates:
        if jaccard(cand.statement, claim.statement) >= DUPLICATE_JACCARD:
            return _apply(
                store,
                Decision(action="duplicate", target_id=cand.id, reason="near-identical wording"),
                claim,
                source,
                report,
            )

    if not candidates:
        decision = Decision(action="add", reason="no similar memories")
    else:
        decision = judge.decide(claim, source.doc_date, candidates)
        if decision.action != "add" and decision.target_id not in {c.id for c in candidates}:
            log.warning("Judge picked unknown target %r; adding as new", decision.target_id)
            decision = Decision(action="add", reason="judge returned an invalid target")
    return _apply(store, decision, claim, source, report)


def _apply(
    store: Store, decision: Decision, claim: ExtractedMemory, source: Source, report: Report
) -> Memory:
    if decision.action == "duplicate":
        store.add_source(decision.target_id, source)
        report.duplicates += 1
        return store.get(decision.target_id)

    new = store.add_memory(claim.kind, claim.subject, claim.statement, source)
    if decision.action == "update":
        store.supersede(decision.target_id, new.id)
        report.updated += 1
    elif decision.action == "conflict":
        store.add_conflict(decision.target_id, new.id, decision.reason)
        new.status = Status.CONFLICTED
        report.conflicts += 1
    else:
        report.added += 1
    return new


# --- orchestration ------------------------------------------------------------


def ingest_document(
    store: Store, extractor: Extractor, judge: Judge, doc: ingest.Document, report: Report
) -> None:
    for chunk in ingest.chunk(doc):
        claims = extractor.extract(chunk.text, doc.path, doc.doc_date)
        report.extracted += len(claims)
        for claim in claims:
            if not is_grounded(claim.evidence, chunk.text):
                report.ungrounded += 1
                log.info("Dropped ungrounded claim from %s: %s", doc.path, claim.statement)
                continue
            source = Source(doc.path, chunk.locator, claim.evidence.strip(), doc.doc_date)
            consolidate(store, judge, claim, source, report)
    store.mark_ingested(doc.path, doc.sha256)
    report.documents += 1


def ingest_path(
    store: Store,
    extractor: Extractor,
    judge: Judge,
    root: Path,
    on_progress: Callable[[str], None] | None = None,
) -> Report:
    """Ingest every supported file under ``root``, oldest document first.

    Oldest-first matters: when a later document changes something, the judge sees
    the old memory already in place and can mark the new one as an *update*.
    Unchanged files (same SHA-256) are skipped, so re-running is cheap.
    """
    report = Report()
    base = root if root.is_dir() else root.parent
    docs = [ingest.load(p, display_path=str(p.relative_to(base))) for p in ingest.discover(root)]
    for doc in sorted(docs, key=lambda d: (d.doc_date, d.path)):
        if store.is_unchanged(doc.path, doc.sha256):
            report.skipped_unchanged += 1
            continue
        if on_progress:
            on_progress(doc.path)
        try:
            ingest_document(store, extractor, judge, doc, report)
        except Exception as exc:  # one bad file should not stop the whole run
            log.exception("Failed to ingest %s", doc.path)
            report.errors.append(f"{doc.path}: {exc}")
    return report
