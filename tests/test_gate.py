import asyncio
import functools

import pytest

from mindpalace import gate, tools
from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import ToolError, merge_entities, save_capture, write_note


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def direct(session):
    async def call(fn):
        return fn(session)
    return call


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def two_proposals(session):
    capture = save_capture(session, "Scaling hits a data wall.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Data binds scaling.",
        relationship_assertions=[
            {"source": "scaling-laws", "target": "data-exhaustion", "type": "contradicts",
             "strength": 8, "description": "the wall"},
        ],
        claim_assertions=[{"subject": "scaling-laws", "text": "compute-optimal needs 20 tokens per parameter"}],
    )
    return session, note["relationship_assertions"][0]["id"], note["claim_assertions"][0]["id"]


def statuses(session):
    return {
        **{r["id"]: r["status"] for r in session.conn.execute("SELECT id, status FROM assertions")},
        **{r["id"]: r["status"] for r in session.conn.execute("SELECT id, status FROM claims")},
    }


def test_review_refuses_a_client_that_cannot_ask(two_proposals, scripted):
    session, _, _ = two_proposals
    with pytest.raises(ToolError, match="cannot ask"):
        run(gate.run_review(scripted(can_ask=False), direct(session), via="chat_review"))


def test_review_with_nothing_waiting_asks_nothing(session, scripted):
    asker = scripted()
    result = run(gate.run_review(asker, direct(session), via="chat_review"))
    assert result["reviewed"] is False and result["pending"] == 0
    assert asker.messages == []


def test_no_to_the_opening_prompt_writes_nothing(two_proposals, scripted):
    session, _, _ = two_proposals
    asker = scripted(confirms=[False])
    result = run(gate.run_review(asker, direct(session), via="chat_review"))
    assert result["reviewed"] is False and result["pending"] == 2
    assert result["next"] == gate.DECLINED_REVIEW_NEXT
    assert "2 proposals are waiting (1 relationship, 1 claim)" in asker.messages[0]
    assert session.decisions.entries() == []


def test_review_confirms_and_dismisses_as_the_person_says(two_proposals, scripted):
    session, rel, claim = two_proposals
    asker = scripted(confirms=[True], choices=[("confirm", "right"), ("dismiss", None)])
    result = run(gate.run_review(asker, direct(session), via="chat_review"))
    assert (result["confirmed"], result["dismissed"], result["skipped"]) == (1, 1, 0)
    assert result["stopped_early"] is False
    assert [(d.assertion, d.action, d.via, d.reason) for d in session.decisions.entries()] == [
        (rel, "confirm", "chat_review", "right"),
        (claim, "dismiss", "chat_review", None),
    ]
    assert statuses(session) == {rel: "confirmed", claim: "dismissed"}


def test_skip_and_stop_leave_items_proposed(two_proposals, scripted):
    session, rel, claim = two_proposals
    asker = scripted(confirms=[True], choices=[("skip", None), ("stop", None)])
    result = run(gate.run_review(asker, direct(session), via="chat_review"))
    assert (result["skipped"], result["stopped_early"]) == (1, True)
    assert session.decisions.entries() == []
    assert statuses(session) == {rel: "proposed", claim: "proposed"}


def test_review_skips_an_item_decided_elsewhere_mid_prompt(two_proposals, scripted):
    session, rel, claim = two_proposals

    def obsidian_dismisses(message):
        if "Relationship" in message:
            session.decisions.append(rel, "dismiss", "obsidian", "obsidian_X")

    asker = scripted(confirms=[True], choices=[("confirm", None), ("confirm", None)],
                     on_ask=obsidian_dismisses)
    result = run(gate.run_review(asker, direct(session), via="chat_review"))
    assert result["already_decided"] == [rel]
    assert result["confirmed"] == 1
    assert [(d.assertion, d.via) for d in session.decisions.entries()] == [
        (rel, "obsidian"), (claim, "chat_review"),
    ]


def test_review_rebuilds_once_at_the_end(two_proposals, scripted, monkeypatch):
    session, _, _ = two_proposals
    rebuilds = []
    real = tools.rebuild_tool
    monkeypatch.setattr(tools, "rebuild_tool", lambda s, *a: rebuilds.append(1) or real(s, *a))
    asker = scripted(confirms=[True], choices=[("confirm", None), ("confirm", None)])
    run(gate.run_review(asker, direct(session), via="chat_review"))
    assert rebuilds == [1]


def test_render_proposal_shows_what_a_person_needs(two_proposals):
    session, _, _ = two_proposals
    rel, claim = tools.review_queue(session)["proposals"]
    text = gate.render_proposal(rel, 1, 2)
    assert text.splitlines()[0] == "(1 of 2) Relationship: scaling-laws —[contradicts]— data-exhaustion"
    assert "Why: the wall" in text and "Strength: 8/10" in text
    assert "scaling-laws:" in text and "data-exhaustion:" in text
    assert "compute-optimal" in gate.render_proposal(claim, 2, 2)


def test_approve_refuses_a_client_that_cannot_ask(session, scripted):
    with pytest.raises(ToolError, match="cannot ask"):
        run(gate.approve(scripted(can_ask=False), direct(session), lambda s: "x", lambda s: {}))


def test_approve_shows_the_summary_and_acts_on_yes(session, scripted):
    asker = scripted(confirms=[True])
    result = run(gate.approve(asker, direct(session), lambda s: "Do the thing.", lambda s: {"ok": 1}))
    assert result == {"ok": 1}
    assert asker.messages == ["Do the thing."]


def test_approve_writes_nothing_on_no(session, scripted):
    acted = []
    with pytest.raises(ToolError, match="declined"):
        run(gate.approve(scripted(confirms=[False]), direct(session),
                         lambda s: "x", lambda s: acted.append(1)))
    assert acted == []


def test_approve_revalidates_after_the_answer(session, scripted):
    capture = save_capture(session, "Two names.")
    write_note(
        session, derived_from=capture["id"], content="n",
        entities=[{"name": "a", "type": "concept", "description": "d"},
                  {"name": "b", "type": "concept", "description": "d"}],
    )

    def meanwhile(message):
        merge_entities(session, "b", "a")  # someone merged the other way

    asker = scripted(confirms=[True], on_ask=meanwhile)
    with pytest.raises(ToolError, match="cycle"):
        run(gate.approve(
            asker, direct(session),
            functools.partial(tools.describe_merge_entities, duplicate="a", canonical="b"),
            functools.partial(tools.merge_entities, duplicate="a", canonical="b", via="chat_approval"),
        ))
    assert session.merges.merges() == {"b": "a"}


def test_render_proposal_shows_the_direction_of_a_directed_edge(session):
    capture = save_capture(session, "x")
    write_note(session, derived_from=capture["id"], content="n", relationship_assertions=[
        {"source": "evidence", "target": "claim", "type": "supports", "description": "backs it"},
    ])
    [proposal] = tools.review_queue(session)["proposals"]
    first = gate.render_proposal(proposal, 1, 1).splitlines()[0]
    assert first == "(1 of 1) Relationship: evidence —[supports]→ claim"
