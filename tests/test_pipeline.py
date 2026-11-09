from conftest import FakeExtractor, FakeJudge, claim

from orgmem import ingest
from orgmem.models import Decision, Source, Status
from orgmem.pipeline import Report, consolidate, ingest_document, ingest_path, is_grounded, jaccard


def src(date="2026-01-01"):
    return Source("d.md", "L1-L1", "quote text", date)


def test_grounding_accepts_real_quotes_with_whitespace_and_smart_quotes():
    text = "We agreed that  production deploys\nare frozen on Fridays."
    assert is_grounded("we agreed that production deploys are frozen", text)
    assert is_grounded("“deploys are frozen on Fridays”", text)


def test_grounding_rejects_invented_or_tiny_quotes():
    text = "We agreed that production deploys are frozen on Fridays."
    assert not is_grounded("deploys are allowed on Fridays", text)
    assert not is_grounded("we", text)


def test_jaccard():
    assert jaccard("Billing uses Postgres", "billing uses postgres.") == 1.0
    assert jaccard("a b", "c d") == 0.0


def test_first_claim_is_added_without_asking_the_judge(store):
    judge = FakeJudge()
    r = Report()
    consolidate(store, judge, claim("Billing uses Postgres.", "x", "billing"), src(), r)
    assert r.added == 1 and judge.seen == []


def test_near_identical_claim_merges_without_llm_call(store):
    judge, r = FakeJudge(), Report()
    first = consolidate(store, judge, claim("Billing uses Postgres.", "x", "billing"), src(), r)
    again = consolidate(store, judge, claim("Billing uses Postgres", "y", "billing"), src(), r)
    assert again.id == first.id and r.duplicates == 1 and judge.seen == []


def test_judge_update_supersedes_old_memory(store):
    r = Report()
    old = consolidate(
        store, FakeJudge(), claim("Friday deploys are frozen.", "x", "deploys"), src(), r
    )
    judge = FakeJudge(Decision(action="update", target_id="FIRST", reason="policy changed"))
    new = consolidate(
        store,
        judge,
        claim("Friday deploys are allowed via canary.", "y", "deploys"),
        src("2026-06-01"),
        r,
    )
    assert store.get(old.id).superseded_by == new.id
    assert r.updated == 1


def test_judge_conflict_flags_both(store):
    r = Report()
    a = consolidate(store, FakeJudge(), claim("Sam owns on-call.", "x", "on-call"), src(), r)
    judge = FakeJudge(Decision(action="conflict", target_id="FIRST", reason="different owners"))
    b = consolidate(store, judge, claim("Dana owns on-call.", "y", "on-call"), src(), r)
    assert {store.get(a.id).status, store.get(b.id).status} == {Status.CONFLICTED}
    assert len(store.open_conflicts()) == 1


def test_invalid_judge_target_falls_back_to_add(store):
    r = Report()
    consolidate(store, FakeJudge(), claim("Sam owns on-call.", "x", "on-call"), src(), r)
    judge = FakeJudge(Decision(action="update", target_id="m_doesnotexist", reason="?"))
    consolidate(store, judge, claim("Dana owns on-call.", "y", "on-call"), src(), r)
    assert r.added == 2 and r.updated == 0


def test_ungrounded_claims_are_dropped(store):
    doc = ingest.from_text("Billing uses Postgres for invoices.", "notes.md", "2026-01-01")
    extractor = FakeExtractor(
        {
            "Billing": [
                claim("Billing uses Postgres.", "Billing uses Postgres", "billing"),
                claim("Billing uses MongoDB.", "Billing uses MongoDB", "billing"),  # invented quote
            ]
        }
    )
    r = Report()
    ingest_document(store, extractor, FakeJudge(), doc, r)
    assert r.extracted == 2 and r.ungrounded == 1 and r.added == 1
    mem = store.all()[0]
    assert mem.sources[0].doc_path == "notes.md" and mem.sources[0].locator == "L1-L1"


def test_ingest_path_is_oldest_first_and_skips_unchanged(store, tmp_path):
    (tmp_path / "2026-06-01-new.md").write_text("Deploys allowed on Fridays now.")
    (tmp_path / "2026-01-01-old.md").write_text("Deploys frozen on Fridays.")
    seen = []
    extractor = FakeExtractor({})
    ingest_path(store, extractor, FakeJudge(), tmp_path, on_progress=seen.append)
    assert seen == ["2026-01-01-old.md", "2026-06-01-new.md"]

    report = ingest_path(store, extractor, FakeJudge(), tmp_path)
    assert report.skipped_unchanged == 2 and extractor.calls == 2


def test_one_failing_document_does_not_stop_the_run(store, tmp_path):
    (tmp_path / "a.md").write_text("boom")
    (tmp_path / "b.md").write_text("fine")

    class Flaky(FakeExtractor):
        def extract(self, text, doc_path, doc_date):
            if "boom" in text:
                raise RuntimeError("API down")
            return []

    report = ingest_path(store, Flaky({}), FakeJudge(), tmp_path)
    assert report.documents == 1 and len(report.errors) == 1
    assert not store.is_unchanged("a.md", "anything")  # failed file will be retried next run
