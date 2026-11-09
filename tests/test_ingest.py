import json

from orgmem import ingest


def test_chunk_keeps_paragraphs_and_line_ranges(tmp_path):
    doc = ingest.from_text("para one\nline two\n\npara two\n\n\npara three", "x", "2026-01-01")
    chunks = ingest.chunk(doc, max_chars=15)
    assert [c.text for c in chunks] == ["para one\nline two", "para two", "para three"]
    assert [c.locator for c in chunks] == ["L1-L2", "L4-L4", "L7-L7"]


def test_chunk_packs_small_paragraphs_together():
    doc = ingest.from_text("a\n\nb\n\nc", "x", "2026-01-01")
    chunks = ingest.chunk(doc)
    assert len(chunks) == 1 and chunks[0].locator == "L1-L5"


def test_date_detection_order(tmp_path):
    front = tmp_path / "notes.md"
    front.write_text("date: 2026-02-03\n\nhello")
    named = tmp_path / "2026-05-06-standup.md"
    named.write_text("hello")
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps({"date": "2026-07-08", "messages": []}))
    assert ingest.load(front).doc_date == "2026-02-03"
    assert ingest.load(named).doc_date == "2026-05-06"
    assert ingest.load(chat).doc_date == "2026-07-08"


def test_chat_json_becomes_speaker_lines(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"title": "T", "messages": [{"author": "Ana", "text": "hi"}]}))
    assert "Ana: hi" in ingest.load(p).text


def test_discover_filters_by_suffix(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.png").write_bytes(b"\x89")
    (tmp_path / ".hidden.md").write_text("x")
    assert [p.name for p in ingest.discover(tmp_path)] == ["a.md"]
