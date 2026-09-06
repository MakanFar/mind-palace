import json

import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import resolve_assertion, save_capture, write_note


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def test_sync_writes_graph_json_with_the_contract_shape(session):
    capture = save_capture(session, "Kaplan 2020 supports scaling laws.")
    note = write_note(
        session, capture["id"], "Body.",
        entities=[{"name": "scaling-laws", "type": "concept", "description": "compute/data/loss"},
                  {"name": "kaplan-2020", "type": "paper", "description": "the paper"}],
        relationship_assertions=[
            {"source": "kaplan-2020", "target": "scaling-laws", "type": "supports", "description": "measures it"},
            {"source": "kaplan-2020", "target": "geforce", "type": "available on", "description": "odd"},
        ],
        claim_assertions=[{"subject": "scaling-laws", "text": "Coined 2020.", "valid_from": "2020"}],
    )
    [typed, untyped] = note["relationship_assertions"]
    resolve_assertion(session, typed["id"], "confirm")

    payload = json.loads(session.paths.graph_json.read_text())
    assert payload["version"] == 1 and payload["generated_at"].endswith("Z")
    slugs = {e["slug"]: e for e in payload["entities"]}
    assert slugs["kaplan-2020"]["type"] == "paper" and slugs["kaplan-2020"]["rank"] == 1
    assert slugs["scaling-laws"]["note_ids"] == [note["id"]]
    assert slugs["scaling-laws"]["description"] == ""  # no page prose written yet
    [edge] = payload["edges"]
    assert edge["source"] == "kaplan-2020" and edge["type"] == "supports" and edge["directed"] is True
    assert edge["traversable"] is True
    [assertion] = edge["assertions"]
    assert assertion["id"] == typed["id"] and assertion["status"] == "confirmed"
    [odd] = payload["untyped"]
    assert odd["id"] == untyped["id"] and odd["proposed_type"] == "available on"
    [claim] = payload["claims"]
    assert claim["valid_from"] == "2020" and claim["subject"] == "scaling-laws"
    assert payload["communities"] == []
    assert payload["vocabulary"][0]["proposed"] == "available-on"
    assert payload["captures"][capture["id"]]["path"].startswith("captures/")
    assert payload["notes"][note["id"]]["path"].startswith("notes/")
    assert set(payload["units"]) == {f"u_{capture['id'].removeprefix('c_')}_0000"}


def test_graph_json_is_rewritten_on_every_sync_and_is_tier_three(session):
    save_capture(session, "one")
    first = session.paths.graph_json.read_text()
    save_capture(session, "two")
    assert session.paths.graph_json.read_text() != first
    session.paths.graph_json.unlink()
    session.resync()
    assert session.paths.graph_json.exists()
