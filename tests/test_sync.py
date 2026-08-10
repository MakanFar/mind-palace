import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.graph.fold import UnknownEdgeTypeError
from mindpalace.index import db
from mindpalace.index import sync as sync_module
from mindpalace.index.sync import has_drift, sync
from mindpalace.models import (
    Capture,
    EntityInstance,
    EntityPage,
    Note,
    RelationshipAssertion,
)
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
def vault(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return paths, VaultStore(paths)


@pytest.fixture
def conn(vault):
    paths, _ = vault
    connection = db.connect(paths.graph_db)
    db.create_schema(connection)
    return connection


def add_note(store, note_id, slug, *, relationships=()):
    note = Note(
        id=note_id,
        derived_from=f"c_{note_id[2:]}",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body=f"Analysis about {slug} and data exhaustion.",
        entities=(EntityInstance(slug, "concept", "A concept."),),
        relationship_assertions=tuple(relationships),
    )
    store.write_note(note, slug)
    return note


def test_sync_populates_graph_tables(conn, vault, config):
    _, store = vault
    add_note(
        store,
        "n_01",
        "scaling-laws",
        relationships=(
            RelationshipAssertion(
                "x_1", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
            ),
        ),
    )
    report = sync(conn, store, config, StubEmbedder(), {"x_1": "confirm"})
    assert report.notes == 1
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
    assert conn.execute("SELECT traversable FROM aggregates").fetchone()[0] == 1


def test_sync_indexes_documents_for_fts(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    hits = list(
        conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("exhaustion",))
    )
    assert ("n_01",) in [tuple(row) for row in hits]


def test_sync_stores_vectors_for_every_indexed_doc(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    vecs = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert docs == vecs > 0


def test_sync_is_idempotent(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    first = sync(conn, store, config, StubEmbedder(), {})
    second = sync(conn, store, config, StubEmbedder(), {})
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == first.captures + 1


def test_malformed_note_becomes_an_issue_without_breaking_the_vault(
    conn, vault, config
):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    (paths.notes / "broken.md").write_text("---\nnot: [closed\n")

    report = sync(conn, store, config, StubEmbedder(), {})

    assert any("broken.md" in issue[0] for issue in report.issues)
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] >= 1
    stored = conn.execute("SELECT COUNT(*) FROM vault_issues").fetchone()[0]
    assert stored == len(report.issues)


def test_malformed_file_stays_searchable(conn, vault, config):
    """Spec §10: a file we cannot parse is still indexed for FTS, not dropped."""
    paths, store = vault
    (paths.notes / "broken.md").write_text(
        "---\nnot: [closed\n\nlithium niobate photonics\n"
    )
    sync(conn, store, config, StubEmbedder(), {})

    hits = [
        row[0]
        for row in conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("niobate",))
    ]
    assert hits == ["notes/broken.md"]


def test_duplicate_assertion_ids_are_reported(conn, vault, config):
    _, store = vault
    shared = RelationshipAssertion(
        "x_dup", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
    )
    add_note(store, "n_01", "scaling-laws", relationships=(shared,))
    add_note(store, "n_02", "chinchilla", relationships=(shared,))

    report = sync(conn, store, config, StubEmbedder(), {})
    assert any(issue[1] == "duplicate_assertion_id" for issue in report.issues)


def test_ambiguous_alias_is_reported_as_a_vault_issue(conn, vault, config):
    """Spec §8.5: two entities claiming the same alias is a vault_issue,
    never silently resolved in favour of one -- aliases explicitly do not
    merge identities."""
    _, store = vault
    store.write_entity_page(
        EntityPage(
            slug="alpha",
            type="concept",
            description="",
            user={"aliases": ["shared-name"]},
        )
    )
    store.write_entity_page(
        EntityPage(
            slug="beta",
            type="concept",
            description="",
            user={"aliases": ["shared-name"]},
        )
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    matches = [issue for issue in report.issues if issue[1] == "ambiguous_alias"]
    assert len(matches) == 1
    _, _, detail = matches[0]
    assert "alpha" in detail and "beta" in detail
    stored = conn.execute(
        "SELECT COUNT(*) FROM vault_issues WHERE kind = 'ambiguous_alias'"
    ).fetchone()[0]
    assert stored == 1


def test_a_repeated_alias_on_one_entity_is_not_a_collision(conn, vault, config):
    """Regression: `_ambiguous_alias_issues` used to append `page.slug` once
    per alias *occurrence* rather than once per *page*, so a single entity
    listing the same alias twice yielded `owners["shared-name"] ==
    ["alpha", "alpha"]` -- length 2, past the skip check -- and reported a
    self-contradictory issue naming only one entity as "claimed by more
    than one entity". One entity repeating its own alias is not a claim by
    more than one entity and must not be reported."""
    _, store = vault
    store.write_entity_page(
        EntityPage(
            slug="alpha",
            type="concept",
            description="",
            user={"aliases": ["shared-name", "shared-name"]},
        )
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert not any(issue[1] == "ambiguous_alias" for issue in report.issues)


def test_a_case_variant_alias_on_one_entity_is_not_a_collision(conn, vault, config):
    """Same bug, reached via the other route into it: two case variants of
    one alias on the same page normalise to the same slug, so they must
    dedupe exactly like a literal repeat does."""
    _, store = vault
    store.write_entity_page(
        EntityPage(
            slug="alpha",
            type="concept",
            description="",
            user={"aliases": ["Shared-Name", "shared-name"]},
        )
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert not any(issue[1] == "ambiguous_alias" for issue in report.issues)


def test_sync_records_the_embedder_in_cache_meta(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    embedder = StubEmbedder()
    sync(conn, store, config, embedder, {})
    assert db.read_meta(conn) == (embedder.model_id, embedder.dim)


def test_sync_maintains_index_md(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})

    text = paths.index_md.read_text()
    assert "n_01" in text
    assert "[[scaling-laws]]" in text


def test_has_drift_detects_an_external_edit(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    target = next(paths.notes.glob("*.md"))
    target.write_text(target.read_text() + "\nEdited in Obsidian.\n")
    assert has_drift(conn, store) is True


def test_has_drift_covers_the_decision_log(conn, vault, config):
    """Decisions are Tier 1 — an external edit must not leave the cache stale."""
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    paths.decisions_log.parent.mkdir(parents=True, exist_ok=True)
    paths.decisions_log.write_text("")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    paths.decisions_log.write_text(
        '{"op":"op_1","ts":"2026-08-01T00:00:00Z","assertion":"x_1",'
        '"action":"confirm","via":"resolve_assertion","reason":null}\n'
    )
    assert has_drift(conn, store) is True


def test_has_drift_covers_the_config(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    paths.mindpalace_md.write_text("---\nschema_version: 1\n---\n")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    paths.mindpalace_md.write_text("---\nschema_version: 1\n---\n\nchanged\n")
    assert has_drift(conn, store) is True


def test_has_drift_detects_a_new_file(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    add_note(store, "n_02", "chinchilla")
    assert has_drift(conn, store) is True


# --- Quarantine behaviour (Task 9's typed FoldError hierarchy) ---
#
# The brief (written before Task 9) has `sync` call `fold(notes, statuses,
# config)` bare. Task 9 made `fold` raise typed exceptions on malformed
# input instead of returning quietly. Calling it bare would let one bad
# note take the whole cache rebuild down, violating spec §10's "one bad
# file must never render the vault unusable." These tests pin the
# quarantine-and-retry behaviour that keeps that promise.


def test_note_with_unknown_edge_type_is_quarantined_but_vault_stays_usable(
    conn, vault, config
):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    add_note(
        store,
        "n_02",
        "chinchilla",
        relationships=(
            RelationshipAssertion(
                "x_bad", "chinchilla", "data-exhaustion", "invented", 5, "nope."
            ),
        ),
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    # the malformed note is excluded from the graph...
    assert report.notes == 1
    issue = next(i for i in report.issues if i[1] == "unknown_edge_type")
    assert "n_02" in issue[0]
    assert "invented" in issue[2]
    assert "x_bad" in issue[2]
    # ...but the good note still made it into the graph...
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM entities WHERE slug = 'scaling-laws'"
        ).fetchone()[0]
        == 1
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM vault_issues").fetchone()[0]
        == len(report.issues)
    )
    # ...and the quarantined note is still findable, not vanished.
    hits = [
        row[0]
        for row in conn.execute(
            "SELECT doc_id, kind FROM docs WHERE docs MATCH ?", ("chinchilla",)
        )
    ]
    assert any("n_02" in doc_id for doc_id in hits)


def test_duplicate_assertion_id_keeps_the_first_note_and_quarantines_the_rest(
    conn, vault, config
):
    _, store = vault
    shared = RelationshipAssertion(
        "x_dup", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
    )
    add_note(store, "n_01", "scaling-laws", relationships=(shared,))
    add_note(store, "n_02", "chinchilla", relationships=(shared,))

    report = sync(conn, store, config, StubEmbedder(), {})

    assert report.notes == 1
    kept = conn.execute(
        "SELECT note_id FROM assertions WHERE id = 'x_dup'"
    ).fetchone()[0]
    assert kept == "n_01"
    issue = next(i for i in report.issues if i[1] == "duplicate_assertion_id")
    assert "n_02" in issue[0]
    assert (
        conn.execute("SELECT COUNT(*) FROM vault_issues").fetchone()[0]
        == len(report.issues)
    )
    hits = [
        row[0]
        for row in conn.execute(
            "SELECT doc_id FROM docs WHERE docs MATCH ?", ("chinchilla",)
        )
    ]
    assert any("n_02" in doc_id for doc_id in hits)


def test_unknown_decision_action_is_quarantined_but_vault_stays_usable(
    conn, vault, config
):
    _, store = vault
    add_note(
        store,
        "n_01",
        "scaling-laws",
        relationships=(
            RelationshipAssertion(
                "x_1", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
            ),
        ),
    )

    report = sync(conn, store, config, StubEmbedder(), {"x_1": "bogus"})

    assert report.notes == 1
    issue = next(i for i in report.issues if i[1] == "unknown_decision_action")
    assert "decisions.jsonl" in issue[0]
    assert "bogus" in issue[2]
    assert "x_1" in issue[2]
    status = conn.execute(
        "SELECT status FROM assertions WHERE id = 'x_1'"
    ).fetchone()[0]
    assert status == "proposed"


def test_unknown_decision_action_does_not_mutate_the_callers_statuses_dict(
    conn, vault, config
):
    _, store = vault
    add_note(
        store,
        "n_01",
        "scaling-laws",
        relationships=(
            RelationshipAssertion(
                "x_1", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
            ),
        ),
    )
    statuses = {"x_1": "bogus"}
    sync(conn, store, config, StubEmbedder(), statuses)
    assert statuses == {"x_1": "bogus"}


def test_fold_quarantine_loop_raises_internal_error_if_it_cannot_shrink(
    monkeypatch, config
):
    """Defence in depth: if a future FoldError subtype (or a bug) is raised
    for a note that quarantine has already excluded, the loop must not spin
    forever -- it must give up loudly within the bounded iteration cap."""
    note = Note(
        id="n_01",
        derived_from="c_01",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="x",
    )

    def broken_fold(notes, statuses, config):
        raise UnknownEdgeTypeError(edge_type="ghost", note_id="n_01", assertion_id="x_1")

    monkeypatch.setattr(sync_module, "fold", broken_fold)

    with pytest.raises(RuntimeError, match="iteration cap"):
        sync_module._fold_with_quarantine(
            [note], {}, config, [], {"n_01": "notes/n_01-scaling-laws.md"}
        )
