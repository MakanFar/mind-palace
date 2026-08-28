import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.graph.fold import fold
from mindpalace.index import db
from mindpalace.models import (
    ClaimAssertion,
    CommunityReport,
    EntityInstance,
    EntityPage,
    Note,
    RelationshipAssertion,
)
from mindpalace.rebuild import community_input_hash, entity_input_hash, rebuild
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={},
    )


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


@pytest.fixture
def conn(store):
    connection = db.connect(store.paths.graph_db)
    db.create_schema(connection)
    return connection


def add_note(store, note_id="n_01", target="data-exhaustion"):
    note = Note(
        id=note_id,
        derived_from="c_01",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="Analysis.",
        entities=(EntityInstance("scaling-laws", "concept", "A concept."),),
        relationship_assertions=(
            RelationshipAssertion(
                f"x_{note_id}", "scaling-laws", target, "contradicts", 8, "because."
            ),
        ),
    )
    store.write_note(note, f"note-{note_id}")
    return note


def add_note_with_claim(store, note_id="n_01", target="data-exhaustion"):
    note = Note(
        id=note_id,
        derived_from="c_01",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="Analysis.",
        entities=(EntityInstance("scaling-laws", "concept", "A concept."),),
        relationship_assertions=(
            RelationshipAssertion(
                f"x_{note_id}", "scaling-laws", target, "contradicts", 8, "because."
            ),
        ),
        claim_assertions=(
            ClaimAssertion(f"k_{note_id}", "scaling-laws", "Scaling laws hold at large N."),
        ),
    )
    store.write_note(note, f"note-{note_id}")
    return note


def test_rebuild_creates_missing_entity_pages_as_stale(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page is not None
    assert page.stale is True
    assert page.description == ""


def test_rebuild_preserves_prose_and_user_overrides(conn, store, config):
    add_note(store)
    store.write_entity_page(
        EntityPage(
            slug="scaling-laws",
            type="concept",
            description="Hand-written prose that must survive.",
            input_hash="sha256:stale",
            stale=False,
            user={"aliases": ["scaling law"], "type": "theme"},
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page.description == "Hand-written prose that must survive."
    assert page.user == {"aliases": ["scaling law"], "type": "theme"}


def test_rebuild_marks_a_page_stale_when_its_inputs_move(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    store.write_entity_page(
        EntityPage(
            slug=page.slug,
            type=page.type,
            description="Now written.",
            generated_from=page.generated_from,
            input_hash=page.input_hash,
            stale=False,
        )
    )
    add_note(store, note_id="n_02", target="chinchilla")

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert store.read_entity_page("scaling-laws").stale is True
    assert report.pages_marked_stale >= 1


def test_editing_a_note_body_marks_its_entity_page_stale(conn, store, config):
    """The regression that identity-hashing missed: the note id never changes,
    so a hash over ids alone would leave the page reading fresh forever."""
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.description = "Written from the original wording."
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace("Analysis.", "Entirely different wording.")
    )

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert store.read_entity_page("scaling-laws").stale is True
    assert report.pages_marked_stale == 1


def test_editing_an_assertion_rationale_marks_the_page_stale(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(note_path.read_text().replace("because.", "for a new reason."))

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_entity_page("scaling-laws").stale is True


def test_editing_an_unconfirmed_assertion_marks_the_page_stale(conn, store, config):
    """An entity's assertions are evidence regardless of review status. Hashing
    only the assertions inside a traversable (confirmed) aggregate would leave
    a page reading fresh forever if its only assertion is still proposed --
    the same silent-staleness regression as identity-hashing a note id, just
    one gate further in."""
    add_note(store)  # x_n_01 is never confirmed in this test
    rebuild(conn, store, config, StubEmbedder(), {})
    page = store.read_entity_page("scaling-laws")
    page.description = "Written while the assertion was still unconfirmed."
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(note_path.read_text().replace("because.", "for a new reason."))

    rebuild(conn, store, config, StubEmbedder(), {})
    assert store.read_entity_page("scaling-laws").stale is True


def test_rebuild_regenerates_the_related_block(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page.related == ["contradicts [[data-exhaustion]]"]


def test_related_block_excludes_untraversable_aggregates(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {})  # nothing confirmed
    assert store.read_entity_page("scaling-laws").related == []


def test_rebuild_never_deletes_community_reports(conn, store, config):
    add_note(store)
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="Prose that no rebuild can recreate.",
            rank=6.0,
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    survivor = store.read_report("g_01")
    assert survivor is not None
    assert survivor.summary == "Prose that no rebuild can recreate."


def test_rebuild_marks_a_report_stale_when_membership_changes(conn, store, config):
    add_note(store)
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, 0, NULL, ?)",
        ("g_01", "scaling-laws,data-exhaustion,chinchilla"),
    )
    conn.commit()
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="Summary.",
            rank=6.0,
            input_hash="sha256:different",
            stale=False,
        )
    )
    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_report("g_01").stale is True
    assert report.reports_marked_stale == 1


# ---- entity_input_hash coverage pins ---------------------------------------
#
# These fields were already in entity_input_hash before this round of fixes;
# they pin behaviour central enough to deserve its own regression test rather
# than riding along inside a broader test.


def test_editing_an_entity_instance_description_marks_the_page_stale(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.description = "Written from the original instance description."
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace("A concept.", "A completely different concept.")
    )

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_entity_page("scaling-laws").stale is True


def test_editing_a_claim_text_marks_the_page_stale(conn, store, config):
    add_note_with_claim(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.description = "Written from the original claim text."
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace(
            "Scaling laws hold at large N.", "Scaling laws break down past 10^26 FLOPs."
        )
    )

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_entity_page("scaling-laws").stale is True


# ---- community_input_hash evidence-breadth fix -----------------------------
#
# community_input_hash must cover the same breadth of evidence as
# entity_input_hash, scoped to a community's members: a report writer can
# `read` a member's raw notes or `get_entity` a member's page before
# composing the report, so entity-instance descriptions, assertion
# rationale/strength regardless of confirmation status, and claim text
# regardless of status are all reachable evidence. Each test below starts a
# report at a correctly-computed, non-stale baseline, edits exactly one class
# of evidence under an unchanged membership list, and checks the report
# flips stale. Each fails against the pre-fix community_input_hash (verified
# by hand -- see the task report).


def test_community_report_flips_stale_when_a_members_instance_description_changes(
    conn, store, config
):
    add_note(store)
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, 0, NULL, ?)",
        ("g_01", "scaling-laws,data-exhaustion"),
    )
    conn.commit()
    store.write_report(
        CommunityReport(
            lineage_id="g_01", level=0, title="Scaling Debate", summary="Summary.", rank=6.0
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    fresh = store.read_report("g_01")
    fresh.stale = False
    store.write_report(fresh)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace("A concept.", "A completely different concept.")
    )

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_report("g_01").stale is True


def test_community_report_flips_stale_when_an_assertion_rationale_changes_under_a_member(
    conn, store, config
):
    add_note(store)
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, 0, NULL, ?)",
        ("g_01", "scaling-laws,data-exhaustion"),
    )
    conn.commit()
    store.write_report(
        CommunityReport(
            lineage_id="g_01", level=0, title="Scaling Debate", summary="Summary.", rank=6.0
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    fresh = store.read_report("g_01")
    fresh.stale = False
    store.write_report(fresh)

    # Strength (and therefore the aggregate's weight/mean_strength) is left
    # untouched -- only the rationale text changes.
    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(note_path.read_text().replace("because.", "for a new reason."))

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_report("g_01").stale is True


def test_community_report_flips_stale_when_an_unconfirmed_claims_text_changes(
    conn, store, config
):
    add_note_with_claim(store)
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, 0, NULL, ?)",
        ("g_01", "scaling-laws,data-exhaustion"),
    )
    conn.commit()
    store.write_report(
        CommunityReport(
            lineage_id="g_01", level=0, title="Scaling Debate", summary="Summary.", rank=6.0
        )
    )
    # k_n_01 is never confirmed -- statuses only confirms the relationship.
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    fresh = store.read_report("g_01")
    fresh.stale = False
    store.write_report(fresh)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace(
            "Scaling laws hold at large N.", "Scaling laws break down past 10^26 FLOPs."
        )
    )

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_report("g_01").stale is True


def test_community_input_hash_is_stable_across_runs(store, config):
    add_note(store)
    notes = list(store.iter_notes())
    notes_by_id = {note.id: note for note in notes}
    tables = fold(notes, {"x_n_01": "confirm"}, config)
    members = ["scaling-laws", "data-exhaustion"]

    first = community_input_hash(members, tables, notes_by_id)
    second = community_input_hash(members, tables, notes_by_id)

    assert first == second


# ---- CRITICAL 1: rebuild() must not call fold() bare -----------------------


def test_rebuild_quarantines_a_note_fold_rejects_instead_of_raising(conn, store, config):
    """rebuild() used to call fold(notes, statuses, config) bare -- a note
    fold() rejects (unknown edge type, duplicate assertion id) took
    rebuild() down, and every write tool calls rebuild()."""
    add_note(store)
    bad = Note(
        id="n_bad",
        derived_from="c_bad",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="A hand-written note with a typo'd edge type.",
        relationship_assertions=(
            RelationshipAssertion(
                "x_bad", "scaling-laws", "data-exhaustion", "invented", 5, "nope."
            ),
        ),
    )
    store.write_note(bad, "bad-note")

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert report.synced == 1  # the good note folded; the bad one was excluded
    issue_kinds = [row[0] for row in conn.execute("SELECT kind FROM vault_issues")]
    assert "unknown_edge_type" in issue_kinds
    # The good note's entity page still gets written -- one bad note does
    # not take the rest of the rebuild down.
    assert store.read_entity_page("scaling-laws") is not None


def test_rebuild_persists_quarantine_issues_even_when_scope_excludes_cache(
    conn, store, config
):
    """`rebuild(scope="related_blocks")` never calls `sync`, which is the
    only other place vault_issues used to get written -- so quarantine
    finding a bad note was correct but invisible to review_queue unless a
    later "all"/"cache" rebuild happened to run."""
    add_note(store)
    bad = Note(
        id="n_bad",
        derived_from="c_bad",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="Bad note.",
        relationship_assertions=(
            RelationshipAssertion(
                "x_bad", "scaling-laws", "data-exhaustion", "invented", 5, "nope."
            ),
        ),
    )
    store.write_note(bad, "bad-note")

    rebuild(
        conn, store, config, StubEmbedder(), {"x_n_01": "confirm"},
        scope="related_blocks",
    )

    issue_kinds = [row[0] for row in conn.execute("SELECT kind FROM vault_issues")]
    assert "unknown_edge_type" in issue_kinds


# ---- IMPORTANT 6: rebuild must not rewrite an unchanged entity page --------


def test_rebuild_does_not_rewrite_an_unchanged_entity_page(conn, store, config):
    """`written` counted every visited entity unconditionally, so a second
    rebuild with nothing changed still rewrote every page (churning git and
    widening the CAS window from Important 5 for no reason)."""
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    page_path = next(store.paths.entities.glob("*.md"))
    before_text = page_path.read_text()
    before_mtime = page_path.stat().st_mtime_ns

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert report.pages_written == 0
    assert page_path.read_text() == before_text
    assert page_path.stat().st_mtime_ns == before_mtime


def test_rebuild_does_not_double_sync_when_nothing_changed(
    conn, store, config, monkeypatch
):
    """`if written or reports_marked: sync(...)` fired a second full sync
    (which re-embeds the whole vault) on every non-empty rebuild, because
    `written` was never actually zero. Once the pages are stable, exactly
    one sync (the scope="all" call) should run per rebuild, not two."""
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    import mindpalace.rebuild as rebuild_module

    calls = []
    original_sync = rebuild_module.sync

    def counting_sync(*args, **kwargs):
        calls.append(1)
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(rebuild_module, "sync", counting_sync)

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert len(calls) == 1


def test_entity_input_hash_is_stable_across_runs(store, config):
    add_note(store)
    notes = list(store.iter_notes())
    notes_by_id = {note.id: note for note in notes}
    tables = fold(notes, {"x_n_01": "confirm"}, config)

    first = entity_input_hash(tables.entities["scaling-laws"], tables, notes_by_id)
    second = entity_input_hash(tables.entities["scaling-laws"], tables, notes_by_id)

    assert first == second
