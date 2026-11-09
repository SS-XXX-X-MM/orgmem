from orgmem.cli import main
from orgmem.export import render_markdown
from orgmem.models import Kind, Source
from orgmem.store import Store


def seed(store):
    s = Source("eng-sync.md", "L4-L5", "q", "2026-03-02")
    store.add_memory(Kind.DECISION, "billing database", "Billing uses Postgres.", s)
    old = store.add_memory(Kind.PROCESS, "deploys", "Friday deploys are frozen.", s)
    new = store.add_memory(Kind.PROCESS, "deploys", "Friday deploys are allowed via canary.", s)
    store.supersede(old.id, new.id)
    a = store.add_memory(Kind.OWNERSHIP, "on-call", "Sam owns on-call.", s)
    b = store.add_memory(Kind.OWNERSHIP, "on-call", "Dana owns on-call.", s)
    store.add_conflict(a.id, b.id, "two owners named on the same day")


def test_export_lists_active_with_citations_and_conflicts(store):
    seed(store)
    md = render_markdown(store)
    assert "Billing uses Postgres. _[eng-sync.md:L4-L5]_" in md
    assert "Friday deploys are allowed via canary." in md
    assert "Friday deploys are frozen." not in md  # superseded memories stay out
    assert "## Needs review" in md and "two owners named on the same day" in md


def test_cli_read_commands(tmp_path, capsys):
    db = tmp_path / "m.db"
    s = Store(db)
    seed(s)
    s.close()

    assert main(["--db", str(db), "search", "billing database"]) == 0
    assert "Billing uses Postgres." in capsys.readouterr().out

    assert main(["--db", str(db), "conflicts"]) == 0
    assert "two owners named on the same day" in capsys.readouterr().out

    out_file = tmp_path / "ORG_MEMORY.md"
    assert main(["--db", str(db), "export", "-o", str(out_file)]) == 0
    assert "# Organisation memory" in out_file.read_text()
