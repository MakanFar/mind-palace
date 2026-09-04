import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import ToolError, ingest_file, read, save_capture, write_note


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path / "vault", init=True, embedder=StubEmbedder()) as opened:
        yield opened


def test_ingest_file_stores_attachment_units_and_returns_a_preview(session, tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("# Title\n\n" + "\n\n".join("para " + "x" * 400 for _ in range(6)))
    result = ingest_file(session, str(src), why="reading")
    assert result["title"] == "Title" and result["parser"] == "markdown" and result["duplicate"] is False
    assert result["units"] >= 2 and result["unit_preview"][0]["id"].startswith("u_")
    assert (session.paths.root / result["attachment"]).read_bytes() == src.read_bytes()
    capture = read(session, result["id"])
    assert capture["kind"] == "capture" and "Title" in capture["text"]
    unit = read(session, result["unit_preview"][0]["id"])
    assert unit["capture"] == result["id"]
    assert "text_unit_ids" in result["next"]


def test_ingesting_the_same_bytes_twice_returns_the_first_capture(session, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("same content")
    first = ingest_file(session, str(src))
    (tmp_path / "b.txt").write_text("same content")
    second = ingest_file(session, str(tmp_path / "b.txt"))
    assert second["id"] == first["id"] and second["duplicate"] is True
    assert len(list(session.paths.attachments.iterdir())) == 1


def test_ingest_file_rejects_a_missing_path(session, tmp_path):
    with pytest.raises(ToolError, match="no such file"):
        ingest_file(session, str(tmp_path / "nope.pdf"))


def test_save_capture_of_a_long_text_records_units_and_keeps_the_text_verbatim(session):
    long = "\n\n".join("sentence " + "y" * 300 for _ in range(8))
    result = save_capture(session, long)
    assert result["units"] >= 2
    assert read(session, f"u_{result['id'].removeprefix('c_')}_0001")["capture"] == result["id"]
    assert read(session, result["id"])["text"] == long


def test_write_note_validates_text_unit_ids(session, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("\n\n".join("sentence " + "z" * 300 for _ in range(8)))
    ingested = ingest_file(session, str(src))
    good = ingested["unit_preview"][0]["id"]
    other = save_capture(session, "unrelated")["id"]
    result = write_note(
        session, ingested["id"], "Body.",
        entities=[
            {"name": "ok", "type": "concept", "description": "d", "text_unit_ids": [good]},
            {"name": "bad", "type": "concept", "description": "d", "text_unit_ids": ["u_nope_0000"]},
            {"name": "wrong", "type": "concept", "description": "d",
             "text_unit_ids": [f"u_{other.removeprefix('c_')}_0000"]},
            {"name": "notalist", "type": "concept", "description": "d", "text_unit_ids": good},
        ],
    )
    assert [e["name"] for e in result["entities"]] == ["ok"]
    assert [d["reason"] for d in result["dropped"]] == ["unknown_text_unit"] * 3
    rows = {tuple(r) for r in session.conn.execute("SELECT item_id, unit_id FROM provenance")}
    assert ("e_ok", good) in rows


def test_ingest_dedup_only_matches_attachments_not_typed_captures(session, tmp_path):
    save_capture(session, "# Report\n\nsame bytes")
    src = tmp_path / "report.md"
    src.write_text("# Report\n\nsame bytes")
    result = ingest_file(session, str(src))
    assert result["duplicate"] is False and result["parser"] == "markdown" and result["attachment"]


def test_a_file_with_no_extractable_text_is_refused_not_stored_empty(session, tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from test_ingest_pdf import pdf_bytes

    src = tmp_path / "scan.pdf"
    src.write_bytes(pdf_bytes([""]))
    with pytest.raises(ToolError, match="no text"):
        ingest_file(session, str(src))
    assert list(session.paths.attachments.iterdir()) == []


def test_reports_can_cite_captures_and_units(session, tmp_path):
    from mindpalace.tools import cluster_tool, write_community_report, propose_relationship, resolve_assertion

    src = tmp_path / "long.txt"
    src.write_text("\n\n".join("sentence " + "q" * 300 for _ in range(8)))
    long = ingest_file(session, str(src))
    short = save_capture(session, "short one")
    unit_of_short = f"u_{short['id'].removeprefix('c_')}_0000"
    note = write_note(session, long["id"], "Body.", entities=[
        {"name": "a", "type": "concept", "description": "d"}, {"name": "b", "type": "concept", "description": "d"}])
    link = propose_relationship(session, "a", "b", "relates-to", "because")
    resolve_assertion(session, link["id"], "confirm")
    clustered = cluster_tool(session, force=True)
    lineage = clustered["communities"][0]["lineage_id"]
    result = write_community_report(
        session, lineage, "T", f"Summary [Data: Captures ({long['id']}); Units ({unit_of_short})]", 5.0,
        [{"summary": f"finding [Data: Entities (e_a)]", "explanation": ""}], [long["id"], unit_of_short],
    )
    assert result["stale"] is False
