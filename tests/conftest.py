"""Deterministic stand-ins for the LLM, so the whole pipeline is testable offline."""

from __future__ import annotations

import pytest

from orgmem.models import Decision, ExtractedMemory, Kind
from orgmem.store import Store


def claim(statement: str, evidence: str, subject: str = "topic", kind: Kind = Kind.FACT):
    return ExtractedMemory(kind=kind, subject=subject, statement=statement, evidence=evidence)


class FakeExtractor:
    """Returns pre-scripted claims for any chunk containing a trigger substring."""

    def __init__(self, script: dict[str, list[ExtractedMemory]]):
        self.script = script
        self.calls = 0

    def extract(self, text, doc_path, doc_date):
        self.calls += 1
        return [c for trigger, claims in self.script.items() if trigger in text for c in claims]


class FakeJudge:
    """Returns queued decisions in order; target_id 'FIRST' means the first candidate."""

    def __init__(self, *decisions: Decision):
        self.queue = list(decisions)
        self.seen: list[list] = []

    def decide(self, new, new_date, candidates):
        self.seen.append(candidates)
        d = self.queue.pop(0)
        if d.target_id == "FIRST":
            d = d.model_copy(update={"target_id": candidates[0].id})
        return d


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()
