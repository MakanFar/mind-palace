"""Tool-level behaviour borrowed from Utopia (docs/decisions/0001)."""

import pytest

from mindpalace.config import add_edge_type, load_config
from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import (
    ToolError,
    adopt_type,
    get_entity,
    graph_stats,
    merge_entities,
    propose_relationship,
    decide,
    resolve_assertion,
    retire_entity,
    review_queue,
    save_capture,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def signed(tmp_path):
    """A vault whose `supports` edge runs paper -> concept only."""
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        pass
    text = (tmp_path / "MINDPALACE.md").read_text()
    text = text.replace(
        "supports: {directed: true, cluster_weight: 1.0}",
        "supports: {directed: true, cluster_weight: 1.0, domain: [paper], range: [concept]}",
    )
    (tmp_path / "MINDPALACE.md").write_text(text)
    with Session(tmp_path, embedder=StubEmbedder()) as opened:
        yield opened


def note(session, **kwargs):
    capture = save_capture(session, kwargs.pop("capture", "A thought."))
    return write_note(session, derived_from=capture["id"], content="Body.", **kwargs)


# ---- 1. untyped proposals, vocabulary, adoption ------------------------------


def test_untyped_assertions_are_counted_as_vocabulary_proposals(session):
    for _ in range(2):
        note(
            session,
            relationship_assertions=[
                {"source": "star-wars", "target": "geforce-now", "type": "Available On",
                 "description": "playable"},
            ],
        )
    queue = review_queue(session)
    [proposal] = queue["vocabulary"]
    assert proposal["kind"] == "edge"
    assert proposal["proposed"] == "available-on"
    assert proposal["count"] == 2
    assert graph_stats(session)["vocabulary_proposals"] == 1
    assert graph_stats(session)["untyped_assertions"] == 2


def test_adopt_type_adds_the_edge_type_and_retypes_waiting_assertions(session):
    first = note(
        session,
        relationship_assertions=[
            {"source": "star-wars", "target": "geforce-now", "type": "available on",
             "description": "playable"},
        ],
    )
    [assertion] = first["relationship_assertions"]
    decide(session, assertion["id"], "confirm", "test")
    assert graph_stats(session)["traversable_aggregates"] == 0

    result = adopt_type(session, "edge", "available on", "available-on", directed=True)
    assert result["adopted"] == "available-on"
    assert result["retyped"] == [assertion["id"]]
    assert "available-on" in load_config(session.paths.mindpalace_md).edge_types
    assert graph_stats(session)["traversable_aggregates"] == 1
    assert graph_stats(session)["untyped_assertions"] == 0
    assert review_queue(session)["vocabulary"] == []
    # The note file itself was never rewritten.
    assert "type: null" in (session.paths.root / first["path"]).read_text()


def test_adopt_type_can_map_onto_an_existing_type_and_be_revoked(session):
    first = note(
        session,
        relationship_assertions=[
            {"source": "a", "target": "b", "type": "backs up", "description": "d"},
        ],
    )
    [assertion] = first["relationship_assertions"]
    adopt_type(session, "edge", "backs up", "supports")
    row = session.conn.execute(
        "SELECT type FROM assertions WHERE id = ?", (assertion["id"],)
    ).fetchone()
    assert row["type"] == "supports"
    adopt_type(session, "edge", "backs up", "supports", action="revoke")
    row = session.conn.execute(
        "SELECT type FROM assertions WHERE id = ?", (assertion["id"],)
    ).fetchone()
    assert row["type"] is None


def test_adopt_type_for_an_entity_type(session):
    note(session, entities=[{"name": "acme", "type": "organisation", "description": "d"}])
    assert get_entity(session, "acme")["type"] == "unknown"
    adopt_type(session, "entity", "organisation", "organisation")
    assert "organisation" in load_config(session.paths.mindpalace_md).entity_types
    assert get_entity(session, "acme")["type"] == "organisation"


def test_adopt_type_validates_its_input(session):
    with pytest.raises(ToolError, match="kind"):
        adopt_type(session, "verb", "x", "x")
    with pytest.raises(ToolError, match="directed"):
        adopt_type(session, "edge", "brand new", "brand-new")


# ---- 2. drop ledger ---------------------------------------------------------


def test_bad_items_are_dropped_individually_and_recorded(session):
    result = note(
        session,
        entities=[
            {"name": "ok", "type": "concept", "description": "fine"},
            {"name": "missing description", "type": "concept"},
            {"name": "this is a whole sentence about something rather long", "type": "concept",
             "description": "x"},
        ],
        relationship_assertions=[
            {"source": "a", "target": "b", "type": "supports", "description": "d", "strength": 99},
            {"source": "a", "target": "a", "type": "supports", "description": "self"},
            {"source": "a", "type": "supports", "description": "no target"},
        ],
        claim_assertions=[
            {"subject": "a", "text": "   "},
            {"subject": "a", "text": "fine"},
        ],
    )
    assert [e["name"] for e in result["entities"]] == ["ok"]
    assert result["relationship_assertions"] == []
    assert len(result["claim_assertions"]) == 1
    reasons = sorted(d["reason"] for d in result["dropped"])
    assert reasons == sorted(
        ["missing_field", "not_an_entity_name", "bad_strength", "self_loop",
         "missing_field", "empty_text"]
    )
    stats = graph_stats(session)["drops"]
    assert stats["total"] == 6
    assert stats["by_reason"]["missing_field"] == 2
    assert any(d["reason"] == "self_loop" for d in review_queue(session)["drops"])
    assert "6 dropped" in result["landed"]


def test_note_level_failures_still_raise(session):
    with pytest.raises(ToolError, match="c_nope"):
        write_note(session, derived_from="c_nope", content="Body.")


# ---- 3. signatures and direction correction ---------------------------------


def test_a_reversed_assertion_is_swapped_and_flagged(signed):
    result = note(
        signed,
        entities=[
            {"name": "scaling-laws", "type": "concept", "description": "d"},
            {"name": "kaplan-2020", "type": "paper", "description": "d"},
        ],
        relationship_assertions=[
            {"source": "scaling-laws", "target": "kaplan-2020", "type": "supports",
             "description": "the paper supports the concept"},
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["pair"] == "kaplan-2020|supports|scaling-laws"
    assert assertion["direction_corrected"] is True
    assert "direction-corrected" in result["landed"]
    assert result["dropped"] == []


def test_an_assertion_valid_in_neither_direction_loses_its_type(signed):
    result = note(
        signed,
        entities=[
            {"name": "alice", "type": "person", "description": "d"},
            {"name": "bob", "type": "person", "description": "d"},
        ],
        relationship_assertions=[
            {"source": "alice", "target": "bob", "type": "supports", "description": "d"},
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["type"] is None
    assert assertion["proposed_type"] == "supports"
    [drop] = result["dropped"]
    assert drop["reason"] == "domain_mismatch"


def test_an_endpoint_of_unknown_type_never_violates(signed):
    result = note(
        signed,
        relationship_assertions=[
            {"source": "mystery", "target": "other", "type": "supports", "description": "d"},
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["type"] == "supports"
    assert assertion["direction_corrected"] is False


def test_endpoint_types_are_resolved_from_the_graph_when_not_in_the_note(signed):
    note(signed, entities=[{"name": "kaplan-2020", "type": "paper", "description": "d"},
                           {"name": "scaling-laws", "type": "concept", "description": "d"}])
    result = propose_relationship(signed, "scaling-laws", "kaplan-2020", "supports", "because")
    assert result["direction_corrected"] is True
    assert result["pair"] == "kaplan-2020|supports|scaling-laws"


# ---- 4. validity and supersession -------------------------------------------


def test_claims_carry_validity_and_get_entity_filters_as_of(session):
    result = note(
        session,
        claim_assertions=[
            {"subject": "acme", "text": "HQ in Beijing.", "valid_from": "2015", "valid_to": "2026-03"},
            {"subject": "acme", "text": "HQ in Shenzhen.", "valid_from": "2026-03-15"},
            {"subject": "acme", "text": "Founded by Wu.", "valid_to": "unknown"},
        ],
    )
    assert len(result["claim_assertions"]) == 3
    claims = get_entity(session, "acme")["claims"]
    by_text = {c["text"]: c for c in claims}
    assert by_text["HQ in Beijing."]["valid_from_precision"] == "year"
    assert by_text["HQ in Beijing."]["valid_to_precision"] == "month"
    assert by_text["Founded by Wu."]["valid_to_precision"] == "unknown"
    as_of = {c["text"] for c in get_entity(session, "acme", as_of="2020-06-01")["claims"]}
    assert as_of == {"HQ in Beijing.", "Founded by Wu."}
    later = {c["text"] for c in get_entity(session, "acme", as_of="2026-04-01")["claims"]}
    assert later == {"HQ in Shenzhen.", "Founded by Wu."}


def test_bad_validity_is_dropped_not_silently_accepted(session):
    result = note(
        session,
        claim_assertions=[
            {"subject": "acme", "text": "a", "valid_from": "March 2026"},
            {"subject": "acme", "text": "b", "valid_from": "2027", "valid_to": "2026"},
            {"subject": "acme", "text": "c", "valid_from": "unknown"},
            {"subject": "acme", "text": "d", "supersedes": "k_nope"},
        ],
    )
    assert result["claim_assertions"] == []
    reasons = sorted(d["reason"] for d in result["dropped"])
    assert reasons == ["bad_validity", "bad_validity", "bad_validity", "unknown_supersedes"]


def test_a_confirmed_correction_supersedes_the_old_claim(session):
    first = note(session, claim_assertions=[{"subject": "acme", "text": "HQ in Beijing."}])
    [old] = first["claim_assertions"]
    decide(session, old["id"], "confirm", "test")
    second = note(
        session,
        claim_assertions=[
            {"subject": "acme", "text": "HQ in Shenzhen.", "supersedes": old["id"]},
        ],
    )
    [new] = second["claim_assertions"]
    assert new["supersedes"] == old["id"]
    statuses = {c["id"]: c["status"] for c in get_entity(session, "acme")["claims"]}
    assert statuses[old["id"]] == "confirmed"
    decide(session, new["id"], "confirm", "test")
    statuses = {c["id"]: c["status"] for c in get_entity(session, "acme")["claims"]}
    assert statuses[old["id"]] == "superseded"
    assert statuses[new["id"]] == "confirmed"


# ---- 5. merges --------------------------------------------------------------


def test_merge_entities_folds_the_duplicate_in_and_is_reversible(session):
    note(session, entities=[{"name": "open-ai", "type": "concept", "description": "the lab"}],
         relationship_assertions=[{"source": "open-ai", "target": "gpt-4", "type": "relates-to",
                                   "description": "built"}])
    note(session, entities=[{"name": "openai", "type": "concept", "description": "the lab"}])
    result = merge_entities(session, "open-ai", "openai", reason="same org")
    assert result["status"] == "merged"
    assert get_entity(session, "openai")["merged_from"] == ["open-ai"]
    assert get_entity(session, "open-ai")["slug"] == "openai"
    slugs = {r["slug"] for r in session.conn.execute("SELECT slug FROM entities")}
    assert "open-ai" not in slugs
    assert graph_stats(session)["merges"] == 1
    merge_entities(session, "open-ai", "openai", action="unmerge")
    slugs = {r["slug"] for r in session.conn.execute("SELECT slug FROM entities")}
    assert "open-ai" in slugs


def test_merge_entities_refuses_self_merges_unknown_slugs_and_cycles(session):
    note(session, entities=[{"name": "a", "type": "concept", "description": "d"},
                            {"name": "b", "type": "concept", "description": "d"}])
    with pytest.raises(ToolError, match="itself"):
        merge_entities(session, "a", "a")
    with pytest.raises(ToolError, match="no entity"):
        merge_entities(session, "ghost", "a")
    merge_entities(session, "a", "b")
    with pytest.raises(ToolError, match="cycle"):
        merge_entities(session, "b", "a")


def test_a_keep_decision_silences_the_similarity_lint(session):
    result = merge_entities(session, "gpt-4", "gpt-4-turbo", action="keep")
    assert result["status"] == "kept"
    assert ("gpt-4", "gpt-4-turbo") in session.merges.kept()


# ---- retirements (docs/decisions/0004) --------------------------------------


def test_retire_entity_removes_an_isolated_entity_and_is_reversible(session):
    note(session, entities=[{"name": "typo", "type": "concept", "description": "d"}])
    assert graph_stats(session)["isolated"] == 1
    result = retire_entity(session, "typo", reason="not a thing")
    assert result == {"slug": "typo", "status": "retired", "reason": "not a thing"}
    slugs = {r["slug"] for r in session.conn.execute("SELECT slug FROM entities")}
    assert "typo" not in slugs
    assert graph_stats(session)["retired"] == 1
    assert graph_stats(session)["isolated"] == 0
    retire_entity(session, "typo", action="restore")
    slugs = {r["slug"] for r in session.conn.execute("SELECT slug FROM entities")}
    assert "typo" in slugs
    assert graph_stats(session)["retired"] == 0


def test_retire_entity_refuses_while_anything_live_touches_it(session):
    [x] = note(
        session,
        entities=[{"name": "a", "type": "concept", "description": "d"}],
        relationship_assertions=[{"source": "a", "target": "b", "type": "relates-to", "description": "d"}],
    )["relationship_assertions"]
    with pytest.raises(ToolError, match="1 proposed relationship.*decide"):
        retire_entity(session, "a")
    decide(session, x["id"], "confirm", "test")
    with pytest.raises(ToolError, match="1 confirmed relationship.*dismiss"):
        retire_entity(session, "a")
    resolve_assertion(session, x["id"], "dismiss")
    # Declared, so still an entity; now isolated, so retirable.
    assert retire_entity(session, "a")["status"] == "retired"
    # Its undeclared endpoint left with the dismissal and is no entity at all.
    with pytest.raises(ToolError, match="no entity"):
        retire_entity(session, "b")


def test_retire_entity_refuses_a_bad_action_and_an_unretired_restore(session):
    note(session, entities=[{"name": "a", "type": "concept", "description": "d"}])
    with pytest.raises(ToolError, match="action must be"):
        retire_entity(session, "a", action="bogus")
    with pytest.raises(ToolError, match="not retired"):
        retire_entity(session, "a", action="restore")


def test_a_live_claim_blocks_retirement_too(session):
    note(
        session,
        entities=[{"name": "a", "type": "concept", "description": "d"}],
        claim_assertions=[{"subject": "a", "text": "Holds."}],
    )
    with pytest.raises(ToolError, match="1 proposed claim"):
        retire_entity(session, "a")


# ---- 7. echo what landed ----------------------------------------------------


def test_write_note_returns_a_landed_summary(session):
    result = note(
        session,
        entities=[
            {"name": "a", "type": "concept", "description": "d"},
            {"name": "b", "type": "widget", "description": "d"},
        ],
        relationship_assertions=[
            {"source": "a", "target": "b", "type": "supports", "description": "d"},
            {"source": "a", "target": "b", "type": "made of", "description": "d"},
        ],
        claim_assertions=[{"subject": "a", "text": "t"}],
    )
    assert result["landed"] == (
        "2 entities (1 untyped), 2 relationships (1 untyped), 1 claim; 0 dropped"
    )
    assert "landed" in result["next"]
    assert result["relationship_assertions"][1]["pair"] == "a|(untyped: made of)|b"


# ---- review findings -------------------------------------------------------


def test_null_and_non_string_fields_are_dropped_not_persisted(session):
    result = note(
        session,
        entities=[
            {"name": None, "type": "concept", "description": "x"},
            {"name": "ok", "type": None, "description": "x"},
            {"name": "ok2", "type": "concept", "description": None},
        ],
        relationship_assertions=[
            {"source": None, "target": "b", "type": "relates-to", "description": "x"},
            {"source": "a", "target": "b", "type": ["relates-to"], "description": "x"},
            {"source": "a", "target": "b", "type": "relates-to", "description": None},
        ],
        claim_assertions=[
            {"subject": None, "text": "x"},
            {"subject": "a", "text": None},
            {"subject": "a", "text": "fine", "valid_from": 2015},
        ],
    )
    assert result["entities"] == []
    assert result["relationship_assertions"] == []
    assert [c["valid_from"] for c in result["claim_assertions"]] == ["2015"]
    assert all(d["reason"] == "bad_value" for d in result["dropped"])
    assert len(result["dropped"]) == 8
    assert review_queue(session)["vocabulary"] == []
    # And the vault still opens and rebuilds afterwards.
    assert graph_stats(session)["drops"]["total"] == 8


def test_merging_entities_with_conflicting_types_does_not_crash_the_lint(session):
    note(session, entities=[{"name": "open-ai", "type": "concept", "description": "d"}])
    note(session, entities=[{"name": "open-ai", "type": "project", "description": "d"}])
    note(session, entities=[{"name": "openai", "type": "project", "description": "d"}])
    merge_entities(session, "open-ai", "openai")
    kinds = {i["kind"] for i in review_queue(session)["vault_issues"]}
    assert "conflicting_entity_type" in kinds
    with Session(session.paths.root, embedder=StubEmbedder()) as reopened:
        assert graph_stats(reopened)["merges"] == 1


def test_a_merged_away_entity_page_is_flagged_and_writes_go_to_the_canonical(session):
    from mindpalace.tools import read, write_entity_description

    note(session, entities=[{"name": "open-ai", "type": "concept", "description": "d"}])
    note(session, entities=[{"name": "openai", "type": "concept", "description": "d"}])
    write_entity_description(session, "open-ai", "OLD")
    merge_entities(session, "open-ai", "openai")
    issues = [i for i in review_queue(session)["vault_issues"] if i["kind"] == "merged_entity_page"]
    assert len(issues) == 1 and "open-ai" in issues[0]["path"]
    write_entity_description(session, "open-ai", "NEW")
    assert get_entity(session, "open-ai")["description"] == "NEW"
    assert read(session, "e_open-ai")["text"] == "NEW"
    docs = {r["doc_id"] for r in session.conn.execute("SELECT doc_id FROM docs WHERE kind = 'entity'")}
    assert "e_open-ai" not in docs


def test_editing_a_description_under_a_merged_name_marks_the_canonical_page_stale(session):
    from mindpalace.tools import rebuild_tool, write_entity_description

    first = note(session, entities=[{"name": "open-ai", "type": "concept", "description": "d1"}])
    note(session, entities=[{"name": "openai", "type": "concept", "description": "d"}])
    merge_entities(session, "open-ai", "openai")
    write_entity_description(session, "openai", "written")
    assert session.store.read_entity_page("openai").stale is False
    path = session.paths.root / first["path"]
    path.write_text(path.read_text().replace("description: d1", "description: d1 edited"))
    rebuild_tool(session)
    assert session.store.read_entity_page("openai").stale is True


def test_adopt_type_validation_failure_leaves_no_pending_op(session):
    with pytest.raises(ToolError, match="directed"):
        adopt_type(session, "edge", "brand new", "brand-new")
    assert session.oplog.pending() == []


def test_a_signature_violated_after_the_fact_is_reported(signed):
    """Enforcement is at write time; a later adoption can move an endpoint
    into a type the signature forbids. That must surface, not vanish."""
    result = note(
        signed,
        entities=[
            {"name": "kaplan-2020", "type": "paper", "description": "d"},
            {"name": "acme", "type": "company", "description": "d"},
        ],
        relationship_assertions=[
            {"source": "kaplan-2020", "target": "acme", "type": "supports", "description": "d"},
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["type"] == "supports"
    adopt_type(signed, "entity", "company", "person")
    issues = [i for i in review_queue(signed)["vault_issues"] if i["kind"] == "signature_violation"]
    assert len(issues) == 1 and "acme" in issues[0]["detail"]


def test_adopting_keeps_the_comments_in_mindpalace_md(session):
    before = session.paths.mindpalace_md.read_text()
    assert "# Optional:" in before
    adopt_type(session, "edge", "available on", "available-on", directed=True)
    adopt_type(session, "entity", "organisation", "organisation")
    after = session.paths.mindpalace_md.read_text()
    assert "# Optional:" in after
    assert "available-on: {directed: true, cluster_weight: 1.0}" in after
    assert "entity_types: [person, concept, paper, project, term, theme, organisation]" in after
