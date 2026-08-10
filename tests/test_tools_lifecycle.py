from datetime import UTC, datetime

import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.ids import new_id
from mindpalace.models import Note, RelationshipAssertion
from mindpalace.session import Session
from mindpalace.tools import (
    ToolError,
    cluster_tool,
    propose_relationship,
    rebuild_tool,
    resolve_assertion,
    review_queue,
    save_capture,
    write_community_report,
    write_entity_description,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def with_assertion(session):
    capture = save_capture(session, "The plateau is about data exhaustion.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply binds scaling.",
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    return session, note["relationship_assertions"][0]["id"]


def test_propose_relationship_creates_a_proposed_assertion(session):
    result = propose_relationship(
        session, "a", "b", "relates-to", "spotted later", strength=6
    )
    assert result["status"] == "proposed"
    assert result["id"].startswith("x_")


def test_propose_relationship_rejects_an_unknown_type(session):
    with pytest.raises(ToolError, match="invented"):
        propose_relationship(session, "a", "b", "invented", "x")


def test_resolve_assertion_confirms(with_assertion):
    session, assertion_id = with_assertion
    assert resolve_assertion(session, assertion_id, "confirm")["status"] == "confirmed"


def test_resolve_assertion_dismisses(with_assertion):
    session, assertion_id = with_assertion
    result = resolve_assertion(session, assertion_id, "dismiss", reason="wrong sense")
    assert result["status"] == "dismissed"


def test_a_dismissal_is_reversible_by_explicit_human_action(with_assertion):
    """Spec §7.3: dismissal is terminal for the proposal loop, not for the human.
    Forcing a whole new assertion to undo a mis-click buys no safety, and every
    flip is recorded in the decision log anyway."""
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "dismiss", reason="wrong sense")
    assert resolve_assertion(session, assertion_id, "confirm")["status"] == "confirmed"
    actions = [entry.action for entry in session.decisions.entries()]
    assert actions == ["dismiss", "confirm"]


def test_previously_dismissed_pairs_are_surfaced_not_suppressed(session):
    """Moved here from Task 15: it needs resolve_assertion to exist."""
    capture = save_capture(session, "A thought.")
    result = write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "chinchilla",
                "type": "contradicts",
                "strength": 5,
                "description": "x",
            }
        ],
    )
    resolve_assertion(
        session,
        result["relationship_assertions"][0]["id"],
        "dismiss",
        reason="different sense",
    )

    payload = save_capture(session, "Another thought about scaling.")
    dismissed = payload["previously_dismissed"]
    assert dismissed[0]["pair"] == "scaling-laws|contradicts|chinchilla"
    assert dismissed[0]["reason"] == "different sense"


def test_resolve_assertion_rejects_an_unknown_action(with_assertion):
    session, assertion_id = with_assertion
    with pytest.raises(ToolError, match="action"):
        resolve_assertion(session, assertion_id, "maybe")


def test_resolve_assertion_rejects_an_unknown_id(session):
    with pytest.raises(ToolError, match="x_ghost"):
        resolve_assertion(session, "x_ghost", "confirm")


def test_review_queue_separates_proposals_from_issues(with_assertion):
    session, _ = with_assertion
    (session.paths.notes / "broken.md").write_text("---\nnot: [closed\n")
    session.resync()

    queue = review_queue(session)
    assert len(queue["proposals"]) == 1
    assert queue["proposals"][0]["pair"].startswith("scaling-laws|contradicts")
    assert any("broken.md" in issue["path"] for issue in queue["vault_issues"])


def test_review_queue_inlines_both_endpoint_snippets(with_assertion):
    """Judging a proposal must not require extra `read` calls."""
    session, _ = with_assertion
    write_entity_description(session, "scaling-laws", "Compute, data, and loss.")
    write_entity_description(session, "data-exhaustion", "Running out of tokens.")

    [proposal] = review_queue(session)["proposals"]
    assert proposal["source_snippet"] == "Compute, data, and loss."
    assert proposal["target_snippet"] == "Running out of tokens."


def test_review_queue_ranks_by_asserted_strength(session):
    capture = save_capture(session, "A thought.")
    write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "a",
                "target": "b",
                "type": "relates-to",
                "strength": 2,
                "description": "weak",
            },
            {
                "source": "c",
                "target": "d",
                "type": "relates-to",
                "strength": 9,
                "description": "strong",
            },
        ],
    )
    strengths = [p["strength"] for p in review_queue(session)["proposals"]]
    assert strengths == [9, 2]


def test_review_queue_drops_resolved_proposals(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    assert review_queue(session)["proposals"] == []


def test_cluster_refuses_below_the_threshold(with_assertion):
    session, _ = with_assertion
    result = cluster_tool(session)
    assert result["clustered"] is False
    assert "threshold" in result["note"]


def test_cluster_with_force_partitions_a_small_graph(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    result = cluster_tool(session, force=True)
    assert result["clustered"] is True
    assert result["communities"]
    assert result["communities"][0]["members"]


def test_write_entity_description_clears_staleness(with_assertion):
    session, _ = with_assertion
    write_entity_description(session, "scaling-laws", "Compute, data, and loss.")
    page = session.store.read_entity_page("scaling-laws")
    assert page.description == "Compute, data, and loss."
    assert page.stale is False


def test_write_community_report_rejects_unresolvable_citations(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="e_ghost"):
        write_community_report(
            session,
            lineage_id,
            title="Bogus",
            summary="Cites something that does not exist.",
            rank=5.0,
            findings=[],
            cites=["e_ghost"],
        )


def test_write_community_report_stores_a_valid_report(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    write_community_report(
        session,
        lineage_id,
        title="Scaling Debate",
        summary="A cluster about scaling limits.",
        rank=6.0,
        findings=[
            {
                "summary": "Data supply dominates",
                "explanation": "Both endpoints agree [Data: Entities (e_scaling-laws)].",
            }
        ],
        cites=["e_scaling-laws"],
    )
    stored = session.store.read_report(lineage_id)
    assert stored.title == "Scaling Debate"
    assert stored.stale is False


def test_write_community_report_requires_every_finding_to_be_grounded(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="no \\[Data"):
        write_community_report(
            session,
            lineage_id,
            title="Ungrounded",
            summary="Reads as evidence but cites nothing.",
            rank=5.0,
            findings=[{"summary": "A confident claim", "explanation": "Trust me."}],
            cites=["e_scaling-laws"],
        )


def test_write_community_report_rejects_a_citation_to_a_missing_note(with_assertion):
    """Note and capture ids used to be waved through without checking."""
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="n_nonexistent"):
        write_community_report(
            session,
            lineage_id,
            title="Bogus",
            summary="Cites a note that was never written.",
            rank=5.0,
            findings=[],
            cites=["n_nonexistent"],
        )


def test_reclustering_preserves_report_lineage(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    before = session.conn.execute("SELECT lineage_id FROM communities").fetchone()[0]
    cluster_tool(session, force=True)
    after = session.conn.execute("SELECT lineage_id FROM communities").fetchone()[0]
    assert before == after


def test_reclustering_marks_an_affected_report_stale_immediately(session):
    """Otherwise global_search serves an obsolete report as `present` until some
    unrelated rebuild happens to run.

    The second piece of evidence is written directly through the store and
    decision log, bypassing `write_note`/`resolve_assertion`. Both of those
    already call `rebuild`, which itself calls `mark_stale_reports` -- so
    going through the public tools here (as the brief's original version of
    this test did) lets that intermediate rebuild catch the staleness
    *before* `cluster_tool` ever runs, leaving `cluster_tool`'s own
    `reports_marked_stale` count at 0 and the test measuring the wrong
    thing. `community_input_hash` deliberately counts every assertion
    touching a member regardless of confirmation status (so a reviewer can
    read a still-proposed rationale), which is exactly why ordinary
    `rebuild` gets there first once real tool calls are involved. Bypassing
    the tools here isolates the specific guarantee this test is named for:
    that `cluster_tool` itself never leaves an evidence-stale report
    reading `stale=False`.
    """
    capture = save_capture(session, "A thought.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "a",
                "target": "b",
                "type": "contradicts",
                "strength": 5,
                "description": "x",
            }
        ],
    )
    resolve_assertion(session, note["relationship_assertions"][0]["id"], "confirm")
    clustered = cluster_tool(session, force=True)
    lineage_id = clustered["communities"][0]["lineage_id"]
    write_community_report(
        session,
        lineage_id,
        title="Pair",
        summary="Written from the original evidence.",
        rank=5.0,
        findings=[],
        cites=["e_a"],
    )
    assert session.store.read_report(lineage_id).stale is False

    second_capture = save_capture(session, "More.")
    second_note = Note(
        id=new_id("n_"),
        derived_from=second_capture["id"],
        created=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        author="llm",
        body="Body two.",
        relationship_assertions=(
            RelationshipAssertion(
                id=new_id("x_"),
                source="a",
                target="b",
                type="contradicts",
                strength=9,
                description="reinforced",
            ),
        ),
    )
    session.store.write_note(second_note, "body-two")
    session.decisions.append(
        second_note.relationship_assertions[0].id, "confirm", "test", "op_test_bypass"
    )
    # Nothing has resynced yet, so the report still reads fresh.
    assert session.store.read_report(lineage_id).stale is False

    result = cluster_tool(session, force=True)
    assert result["reports_marked_stale"] == 1
    assert session.store.read_report(lineage_id).stale is True


def test_rebuild_tool_rejects_an_unknown_scope(session):
    """The brief lists `rebuild_tool` among the interfaces this task
    produces but did not include a test for it (unlike every other
    function here). `rebuild()` itself already has its own dedicated test
    module; this only needs to confirm the tools.py wrapper validates its
    input and passes the session's pieces through correctly."""
    with pytest.raises(ToolError, match="scope"):
        rebuild_tool(session, scope="nonsense")


def test_rebuild_tool_reports_what_it_did(with_assertion):
    session, _ = with_assertion
    result = rebuild_tool(session)
    assert result["scope"] == "all"
    assert result["notes_synced"] == 1
