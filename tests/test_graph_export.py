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


def test_a_missing_graph_json_is_regenerated_on_open(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as first:
        save_capture(first, "one")
        first.paths.graph_json.unlink()
    with Session(tmp_path, embedder=StubEmbedder()) as second:
        assert second.paths.graph_json.exists()


def test_cluster_refreshes_graph_json(session):
    from mindpalace.tools import cluster_tool, propose_relationship

    write_note(session, save_capture(session, "x")["id"], "Body.", entities=[
        {"name": "a", "type": "concept", "description": "d"}, {"name": "b", "type": "concept", "description": "d"}])
    link = propose_relationship(session, "a", "b", "relates-to", "because")
    resolve_assertion(session, link["id"], "confirm")
    assert json.loads(session.paths.graph_json.read_text())["communities"] == []
    cluster_tool(session, force=True)
    communities = json.loads(session.paths.graph_json.read_text())["communities"]
    assert communities and set(communities[0]["members"]) >= {"a", "b"}


def test_a_decision_appended_during_sync_is_still_drift(tmp_path):
    """The Obsidian plugin appends without the vault lock. A line landing while
    sync is embedding must not be hashed as already folded."""
    from mindpalace.index.sync import has_drift

    class SlowEmbedder(StubEmbedder):
        def __init__(self, hook):
            super().__init__()
            self.hook = hook

        def embed(self, texts):
            self.hook()
            return super().embed(texts)

    state = {"session": None, "fired": False}

    def append_during_embed():
        s = state["session"]
        if s is None or state["fired"]:
            return
        state["fired"] = True
        s.paths.decisions_log.parent.mkdir(exist_ok=True)
        with s.paths.decisions_log.open("a") as f:
            f.write('{"action":"confirm","assertion":"x_late","op":"obsidian_1","reason":null,"ts":"t","via":"obsidian"}\n')

    with Session(tmp_path, init=True, embedder=SlowEmbedder(append_during_embed)) as s:
        state["session"] = s
        save_capture(s, "one")  # the resync inside fires the hook mid-embed
        assert state["fired"]
        assert has_drift(s.conn, s.store)
