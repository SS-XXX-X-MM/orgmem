import pytest

from orgmem.models import Kind, Source, Status
from orgmem.store import fts_query


def src(doc="d.md", quote="some quote", date="2026-01-01"):
    return Source(doc, "L1-L2", quote, date)


def test_fts_query_quotes_words_so_user_input_cannot_break_syntax():
    assert fts_query('deploy AND "friday" OR x') == '"and" OR "deploy" OR "friday"'
    assert fts_query("a b") == ""


def test_add_and_search_ranks_relevant_first(store):
    store.add_memory(Kind.FACT, "billing", "Billing uses Postgres.", src())
    store.add_memory(Kind.FACT, "clusters", "The EU cluster runs in eu-west-1.", src())
    results = store.search("which database does billing use")
    assert results[0].statement == "Billing uses Postgres."


def test_duplicate_source_is_ignored(store):
    m = store.add_memory(Kind.FACT, "x", "Statement.", src())
    store.add_source(m.id, src())
    store.add_source(m.id, src(doc="other.md"))
    assert len(store.get(m.id).sources) == 2


def test_supersede_hides_from_search_and_keeps_history(store):
    old = store.add_memory(Kind.PROCESS, "deploys", "Friday deploys are frozen.", src())
    new = store.add_memory(Kind.PROCESS, "deploys", "Friday deploys are allowed.", src())
    store.supersede(old.id, new.id)
    assert [m.id for m in store.search("friday deploys")] == [new.id]
    assert [m.id for m in store.history(new.id)] == [old.id]


def test_conflict_flags_both_and_resolve_keeps_one(store):
    a = store.add_memory(Kind.OWNERSHIP, "on-call", "Sam owns on-call.", src())
    b = store.add_memory(Kind.OWNERSHIP, "on-call", "Dana owns on-call.", src())
    c = store.add_conflict(a.id, b.id, "different owners")
    assert store.get(a.id).status is Status.CONFLICTED
    store.resolve_conflict(c.id, keep_id=b.id)
    assert store.get(b.id).status is Status.ACTIVE
    assert store.get(a.id).status is Status.SUPERSEDED
    assert store.open_conflicts() == []


def test_resolve_rejects_unrelated_memory(store):
    a = store.add_memory(Kind.FACT, "x", "A.", src())
    b = store.add_memory(Kind.FACT, "x", "B.", src())
    other = store.add_memory(Kind.FACT, "y", "C.", src())
    c = store.add_conflict(a.id, b.id, "r")
    with pytest.raises(ValueError):
        store.resolve_conflict(c.id, keep_id=other.id)


def test_unchanged_document_detection(store):
    assert not store.is_unchanged("a.md", "h1")
    store.mark_ingested("a.md", "h1")
    assert store.is_unchanged("a.md", "h1")
    assert not store.is_unchanged("a.md", "h2")
