import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.models import Capture
from mindpalace.session import Session
from mindpalace.tools import (
    cluster_tool,
    get_entity,
    graph_stats,
    neighbors,
    rebuild_tool,
    decide,
    review_queue,
    save_capture,
    search_global,
    search_local,
    write_community_report,
    write_entity_description,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


DISTRACTOR_CAPTURES = (
    "The sourdough starter needs feeding twice a day.",
    "Cold front moving through the valley tonight.",
    "Tomatoes need staking before the weekend.",
    "The commuter train was delayed again this morning.",
    "Simmer the stock for two hours before straining.",
    "Remember to renew the passport before the trip.",
    "The new album from that band drops on Friday.",
    "The bike chain keeps slipping on the middle ring.",
    "Book the ferry crossing early to save money.",
)


def seed(session):
    """Two linked notes, one confirmed relationship.

    Also seeds nine unrelated distractor captures. SQLite FTS5's bm25() uses
    the unsmoothed idf = ln((N - df + 0.5) / (df + 0.5)); in a tiny corpus
    where the query terms appear in roughly half the documents, idf lands at
    ~0 and every match scores ~0 regardless of relevance — a degenerate
    corpus, not a degenerate query (see tests/test_retrieve.py, which
    documents the same effect and pads to ~10 documents for the same reason).
    Without the distractors, `search_local` can never clear
    `abstain_bm25_floor` here and every assertion depending on a non-empty
    `hits` list fails for a reason that has nothing to do with the code under
    test.
    """
    for text in DISTRACTOR_CAPTURES:
        save_capture(session, text)
    first = save_capture(session, "The plateau talk is mostly about data exhaustion.")
    note = write_note(
        session,
        derived_from=first["id"],
        content="Data supply binds scaling, not architecture.",
        entities=[
            {"name": "scaling-laws", "type": "concept", "description": "Compute vs loss."},
            {"name": "data-exhaustion", "type": "concept", "description": "Running out."},
        ],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "Plateau is a supply constraint, not a ceiling.",
            }
        ],
        claim_assertions=[
            {"subject": "scaling-laws", "text": "The plateau reflects data exhaustion."}
        ],
    )
    decide(session, note["relationship_assertions"][0]["id"], "confirm", "test")
    return note


def snapshot(session) -> dict:
    """Everything a query could observe, for the derivability invariant."""
    def rows(sql):
        return [tuple(row) for row in session.conn.execute(sql)]

    return {
        "entities": rows("SELECT slug, type, rank FROM entities ORDER BY slug"),
        "aggregates": rows(
            "SELECT key, weight, mean_strength, traversable FROM aggregates ORDER BY key"
        ),
        "assertions": rows("SELECT id, status FROM assertions ORDER BY id"),
        "claims": rows("SELECT id, status FROM claims ORDER BY id"),
        "docs": rows("SELECT doc_id, kind FROM docs ORDER BY doc_id"),
        "search": [
            hit["id"] for hit in search_local(session, "data exhaustion")["hits"]
        ],
        "neighbours": neighbors(session, "e_scaling-laws")["neighbours"],
        "entity_related": [
            (page.slug, tuple(sorted(page.related)))
            for page in sorted(
                session.store.iter_entity_pages(), key=lambda p: p.slug
            )
        ],
    }


def test_full_cycle_from_capture_to_global_answer(session):
    seed(session)

    assert review_queue(session)["proposals"][0]["kind"] == "claim"
    assert search_local(session, "data exhaustion")["hits"]
    assert neighbors(session, "e_scaling-laws")["neighbours"][0]["slug"] == "data-exhaustion"
    assert get_entity(session, "scaling-laws")["rank"] == 1

    clustered = cluster_tool(session, force=True)
    assert clustered["clustered"] is True
    lineage_id = clustered["communities"][0]["lineage_id"]

    write_community_report(
        session,
        lineage_id,
        title="Scaling and Data Supply",
        summary="A tension between architectural and data-supply accounts.",
        rank=7.0,
        findings=[
            {
                "summary": "Data supply is the binding constraint",
                "explanation": "Both notes agree [Data: Entities (e_scaling-laws)].",
            }
        ],
        cites=["e_scaling-laws", "e_data-exhaustion"],
    )

    answer = search_global(session, "what are the themes here?")
    assert answer["available"] is True
    assert answer["communities"][0]["report"] == "present"


def test_derived_cache_is_rebuildable_from_source(session):
    """The invariant: Tier 3 is a pure function of Tier 1."""
    seed(session)
    before = snapshot(session)

    session.conn.close()
    session.paths.graph_db.unlink()
    for page_path in session.paths.entities.glob("*.md"):
        text = page_path.read_text()
        head, _, _ = text.partition("<!-- mindpalace:related -->")
        page_path.write_text(head)

    from mindpalace.index import db

    session.conn = db.connect(session.paths.graph_db)
    db.create_schema(session.conn)
    rebuild_tool(session, "all")

    assert snapshot(session) == before


def test_rebuild_never_destroys_generated_prose(session):
    """Tier 2 is replaceable, not derivable — rebuild must not touch it."""
    seed(session)
    write_entity_description(session, "scaling-laws", "Prose no rebuild can recreate.")
    clustered = cluster_tool(session, force=True)
    write_community_report(
        session,
        clustered["communities"][0]["lineage_id"],
        title="Scaling and Data Supply",
        summary="Report prose that must survive.",
        rank=7.0,
        findings=[],
        cites=["e_scaling-laws"],
    )

    rebuild_tool(session, "all")

    assert (
        session.store.read_entity_page("scaling-laws").description
        == "Prose no rebuild can recreate."
    )
    reports = list(session.store.iter_reports())
    assert reports[0].summary == "Report prose that must survive."


def test_user_overrides_survive_a_full_rebuild(session):
    seed(session)
    page = session.store.read_entity_page("scaling-laws")
    page.user = {"aliases": ["scaling law"], "type": "theme"}
    session.store.write_entity_page(page)

    rebuild_tool(session, "all")

    survivor = session.store.read_entity_page("scaling-laws")
    assert survivor.user == {"aliases": ["scaling law"], "type": "theme"}
    assert get_entity(session, "Scaling Law")["slug"] == "scaling-laws"


def test_reprocessing_never_double_counts(session):
    seed(session)
    weight_before = session.conn.execute(
        "SELECT weight FROM aggregates"
    ).fetchone()[0]

    rebuild_tool(session, "all")
    rebuild_tool(session, "all")

    assert session.conn.execute("SELECT weight FROM aggregates").fetchone()[0] == (
        weight_before
    )


def test_an_external_edit_is_healed_on_reopen(session, tmp_path):
    seed(session)
    root = session.paths.root
    orphan = Capture(
        id="c_01HQZZZZZZ",
        created="2026-08-09T09:00:00Z",
        source="manual",
        why=None,
        text="Written straight into the vault by another tool.",
    )
    (root / "captures" / "2026-08-09-0900-ZZZZZZ.md").write_text(
        "---\nid: c_01HQZZZZZZ\ncreated: '2026-08-09T09:00:00Z'\nsource: manual\n---\n\n"
        + orphan.text
        + "\n"
    )
    session.close()

    with Session(root, embedder=StubEmbedder()) as reopened:
        found = reopened.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE doc_id = ?", ("c_01HQZZZZZZ",)
        ).fetchone()[0]
        assert found == 1


def test_a_malformed_file_does_not_disable_the_vault(session):
    seed(session)
    (session.paths.notes / "broken.md").write_text("---\nnot: [closed\n")
    session.resync()

    assert search_local(session, "data exhaustion")["hits"]
    assert any(
        issue["kind"] == "malformed_note" for issue in review_queue(session)["vault_issues"]
    )


def test_graph_stats_answers_is_global_search_worth_trying(session):
    seed(session)
    stats = graph_stats(session)
    assert stats["clustering"]["active"] is False
    assert stats["clustering"]["remaining"] > 0
    assert search_global(session, "themes?")["available"] is False


# ---- CRITICAL 1 / 2: one bad file must never render the vault unusable ---
#
# `test_a_malformed_file_does_not_disable_the_vault` above only covers the
# one path that already worked: `sync`/`resync` quarantining an unparseable
# *note*. `rebuild()` (called by every write tool) used to call `fold()`
# bare -- defeating that quarantine one layer up -- and `VaultStore` used to
# raise bare on a malformed *entity page* too. These pin both gaps at the
# tool surface: the whole write path, not just `resync`, must keep working.


def test_a_hand_written_note_that_fold_rejects_does_not_disable_the_vault(session):
    """CRITICAL 1: a note with a typo'd edge type (or a duplicate assertion
    id from a git merge, or an edge type MINDPALACE.md stopped declaring)
    parses fine but makes `fold()` raise. `sync`/`resync` already
    quarantined this correctly; `rebuild()` -- and therefore every write
    tool -- used to call `fold()` a second time, bare, and take the whole
    tool surface down with it."""
    seed(session)
    (session.paths.notes / "n_badedge-hand.md").write_text(
        "---\n"
        "id: n_badedge\n"
        "derived_from: c_none\n"
        "created: '2026-08-01T00:00:00Z'\n"
        "author: llm\n"
        "relationship_assertions:\n"
        "  - id: x_bad\n"
        "    source: scaling-laws\n"
        "    target: data-exhaustion\n"
        "    type: nonsense-type\n"
        "    strength: 5\n"
        "    description: bad edge type\n"
        "---\n\n"
        "Body text.\n"
    )

    # The write surface -- not just resync -- must still respond.
    second = save_capture(session, "A completely unrelated capture about tomatoes.")
    write_note(session, derived_from=second["id"], content="Unrelated analysis.")

    assert any(
        issue["kind"] == "unknown_edge_type"
        for issue in review_queue(session)["vault_issues"]
    )


def test_a_hand_written_bad_entity_page_does_not_disable_the_vault(session):
    """CRITICAL 2: dropping a malformed `entities/*.md` file into the vault
    -- bad YAML, or valid YAML missing a required key like `type` -- used to
    raise straight out of `VaultStore`, bare, into `sync`/`resync` and
    therefore into `rebuild`. Every write tool then failed too."""
    seed(session)
    (session.paths.entities / "hand-made.md").write_text("---\ntitle: oops\n---\n")

    session.resync()
    assert any(
        issue["kind"] == "malformed_entity_page"
        for issue in review_queue(session)["vault_issues"]
    )

    # The write surface must still respond, not raise.
    second = save_capture(session, "Another unrelated capture about bicycles.")
    write_note(session, derived_from=second["id"], content="More analysis.")
    assert search_local(session, "data exhaustion")["hits"]


# ---- IMPORTANT 3: hand-written content below the related block survives --


def test_rebuild_preserves_a_hand_written_section_below_the_related_block(session):
    """The finding's exact reproduction: append a section below the
    machine-owned block, call rebuild, and it used to be gone -- `rebuild`
    runs on every write tool, so this was a live trap, not a corner case.

    Important 6's unchanged-page skip means `rebuild` never rewrites a page
    whose type/generated_from/input_hash/stale/related haven't moved -- so
    appending trailing text alone (which affects none of those) makes
    `rebuild` skip the page entirely, and the trailing text would trivially
    "survive" by never being touched at all. That made this test pass even
    with the trailing re-emit deleted (confirmed by the scoped re-review's
    mutation testing). Editing the note's assertion rationale in place moves
    this entity's `input_hash`, which IS one of the fields the unchanged-check
    compares, forcing an actual rewrite -- and asserting `pages_written >= 1`
    pins that the rewrite really happened rather than being skipped again.
    """
    seed(session)
    page_path = next(session.paths.entities.glob("*.md"))
    page_path.write_text(
        page_path.read_text() + "\n## My own notes\n\nHand written, must survive.\n"
    )

    note_path = next(session.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace(
            "Plateau is a supply constraint, not a ceiling.",
            "A materially different rationale that moves the input hash.",
        )
    )

    report = rebuild_tool(session, "all")

    assert report["pages_written"] >= 1
    assert "Hand written, must survive." in page_path.read_text()
