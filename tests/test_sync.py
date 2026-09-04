import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.graph.fold import UnknownEdgeTypeError
from mindpalace.index import db
from mindpalace.index import sync as sync_module
from mindpalace.ids import slugify
from mindpalace.index.sync import has_drift, sync
from mindpalace.models import (
    Capture,
    CommunityReport,
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

    def broken_fold(notes, statuses, config, **_overlays):
        raise UnknownEdgeTypeError(edge_type="ghost", note_id="n_01", assertion_id="x_1")

    monkeypatch.setattr(sync_module, "fold", broken_fold)

    with pytest.raises(RuntimeError, match="iteration cap"):
        sync_module._fold_with_quarantine(
            [note], {}, config, [], {"n_01": "notes/n_01-scaling-laws.md"}
        )


# --- CRITICAL 2: malformed Tier 2 files must degrade like Tier 1 does ------
#
# `iter_entity_pages`/`iter_reports` used to raise (FrontMatterError for bad
# YAML, a bare KeyError for valid YAML missing a required key) straight out
# of `sync`, exactly the failure mode Task 9's quarantine was built to
# prevent for notes. These pin the Tier-2 counterpart of the tests above.


def test_a_malformed_entity_page_becomes_an_issue_and_stays_searchable(
    conn, vault, config
):
    """Spec §10 applies to a hand-edited entity page exactly as it does to
    a hand-edited note."""
    paths, store = vault
    (paths.entities / "hand-made.md").write_text(
        "---\nnot: [closed\n\nlithium niobate photonics\n"
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert any(issue[1] == "malformed_entity_page" for issue in report.issues)
    hits = [
        row[0]
        for row in conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("niobate",))
    ]
    assert hits == ["entities/hand-made.md"]


def test_an_entity_page_missing_a_required_key_becomes_an_issue(conn, vault, config):
    """The exact reproduction from the finding: valid YAML, but missing
    'type', used to die with a bare `KeyError: 'type'` rather than being
    recorded and skipped.

    `_load_entity_pages` deliberately catches broad `Exception` (so a
    future, unanticipated failure mode still degrades rather than raising),
    which means asserting only the issue *kind* here doesn't pin the
    contract this finding is actually about: `str(KeyError('type'))` is
    `"'type'"`, which still contains the substring "type", so a plain
    `"type" in detail` check -- or no check on the detail at all -- would
    keep passing even if the bare `KeyError` this finding names were
    reinstated. Assert the specific message `read_entity_page` raises
    instead, which only the `FrontMatterError` path produces.
    """
    paths, store = vault
    (paths.entities / "hand-made.md").write_text("---\ntitle: oops\n---\n")

    report = sync(conn, store, config, StubEmbedder(), {})

    issue = next(i for i in report.issues if i[1] == "malformed_entity_page")
    assert "missing required front-matter key 'type'" in issue[2]


def test_a_malformed_community_report_becomes_an_issue_and_stays_searchable(
    conn, vault, config
):
    paths, store = vault
    (paths.communities / "g_bad.md").write_text(
        "---\nnot: [closed\n\nlithium niobate photonics\n"
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert any(issue[1] == "malformed_report" for issue in report.issues)
    hits = [
        row[0]
        for row in conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("niobate",))
    ]
    assert hits == ["communities/g_bad.md"]


# --- IMPORTANT 8: has_drift must cover Tier-2 files feeding the cache -----


def test_has_drift_detects_an_edit_to_an_entity_page(conn, vault, config):
    """`entities/` fed `docs`/`vectors` but was never in
    `_iter_source_files`, so an Obsidian edit to an entity description --
    or a hand-added `user.aliases` -- was invisible to `has_drift`, and
    `heal()` could never notice it."""
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    store.write_entity_page(
        EntityPage(slug="scaling-laws", type="concept", description="Original.")
    )
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    page_path = next(store.paths.entities.glob("*.md"))
    page_path.write_text(page_path.read_text() + "\nEdited in Obsidian.\n")
    assert has_drift(conn, store) is True


def test_has_drift_detects_an_edit_to_a_community_report(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    store.write_report(
        CommunityReport(lineage_id="g_01", level=0, title="T", summary="S", rank=5.0)
    )
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    report_path = next(store.paths.communities.glob("*.md"))
    report_path.write_text(report_path.read_text() + "\nEdited in Obsidian.\n")
    assert has_drift(conn, store) is True


def test_near_duplicate_entities_are_reported_as_a_vault_issue(conn, vault, config):
    """Spec §10 defers automatic near-duplicate detection to a Phase-1 lint.

    Detection only: the pair is surfaced through `review_queue` for a human
    to reconcile, never merged (spec §8.5).
    """
    _, store = vault
    add_note(store, "n_01", "openai")
    add_note(store, "n_02", "openai inc")

    report = sync(conn, store, config, StubEmbedder(), {})

    matches = [issue for issue in report.issues if issue[1] == "near_duplicate_entity"]
    assert len(matches) == 1
    _, _, detail = matches[0]
    assert "openai" in detail and "openai-inc" in detail
    stored = conn.execute(
        "SELECT COUNT(*) FROM vault_issues WHERE kind = 'near_duplicate_entity'"
    ).fetchone()[0]
    assert stored == 1


def test_a_near_duplicate_already_declared_as_an_alias_is_not_reported(
    conn, vault, config
):
    """Once reconciled by hand the lint must go quiet, or it trains its
    reader to ignore it. Catches `sync` folding the graph without passing
    the entity pages' declared aliases through to the lint.
    """
    _, store = vault
    add_note(store, "n_01", "openai")
    add_note(store, "n_02", "openai inc")
    store.write_entity_page(
        EntityPage(
            slug="openai",
            type="concept",
            description="",
            user={"aliases": ["openai inc"]},
        )
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert not any(issue[1] == "near_duplicate_entity" for issue in report.issues)


def typed_note(store, note_id, name, entity_type, *, relationships=()):
    """`add_note` hardcodes a valid type; these tests need a chosen one."""
    note = Note(
        id=note_id,
        derived_from=f"c_{note_id[2:]}",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body=f"Analysis about {name}.",
        entities=(EntityInstance(name, entity_type, "A thing."),),
        relationship_assertions=tuple(relationships),
    )
    store.write_note(note, slugify(name))
    return note


def test_an_entity_type_absent_from_config_is_reported_as_a_vault_issue(
    conn, vault, config
):
    """`fold` accepts any string in `entities.type`; only `write_note`
    validates, and only for notes it wrote. A hand-edited note gets no
    check at all today.
    """
    _, store = vault
    typed_note(store, "n_01", "scaling", "concpet")

    report = sync(conn, store, config, StubEmbedder(), {})

    matches = [issue for issue in report.issues if issue[1] == "unknown_entity_type"]
    assert len(matches) == 1
    path, _, detail = matches[0]
    assert "scaling" in path
    assert "concpet" in detail
    stored = conn.execute(
        "SELECT COUNT(*) FROM vault_issues WHERE kind = 'unknown_entity_type'"
    ).fetchone()[0]
    assert stored == 1


def test_an_entity_only_referenced_by_an_assertion_is_reported(conn, vault, config):
    """`data-exhaustion` enters the graph with the `unknown` sentinel type
    and no page-worthy declaration behind it.
    """
    _, store = vault
    typed_note(
        store,
        "n_01",
        "scaling",
        "concept",
        relationships=(
            RelationshipAssertion(
                "x_1", "scaling", "data exhaustion", "contradicts", 8, "because."
            ),
        ),
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    matches = [i for i in report.issues if i[1] == "reference_only_entity"]
    assert [i[0].endswith("data-exhaustion.md") for i in matches] == [True]
    assert "data-exhaustion" in matches[0][2]


def test_a_type_finding_from_a_quarantined_note_is_not_reported(conn, vault, config):
    """The note never made it into the graph, so complaining about the type
    of an entity it declared points the user at a second problem that will
    disappear the moment they fix the first one.
    """
    _, store = vault
    typed_note(
        store,
        "n_01",
        "scaling",
        "concpet",
        relationships=(
            RelationshipAssertion("x_1", "scaling", "other", "nonesuch", 8, "because."),
        ),
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert any(issue[1] == "unknown_edge_type" for issue in report.issues)
    assert not any(issue[1] == "unknown_entity_type" for issue in report.issues)


def test_a_hand_edited_note_with_an_unquoted_timestamp_still_syncs(conn, vault, config):
    """Spec §10: one bad file must never render the vault unusable.

    An unquoted `created:` is valid YAML for a datetime, so this file used
    to reach `fold` with a datetime among strings and take the whole sync
    down with a TypeError -- which is not a FoldError, so the quarantine
    pass could not catch it either.
    """
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    (store.paths.notes / "handwritten.md").write_text(
        "---\n"
        "id: n_zz\n"
        "created: 2026-08-02T00:00:00Z\n"
        "author: human\n"
        "entities:\n"
        "  - name: Data Exhaustion\n"
        "    type: concept\n"
        "    description: A limit.\n"
        "---\n"
        "\n"
        "Hand-written in Obsidian.\n"
    )

    report = sync(conn, store, config, StubEmbedder(), {})

    assert report.notes == 2
    assert not any(issue[1] == "malformed_note" for issue in report.issues)
    slugs = {row[0] for row in conn.execute("SELECT slug FROM entities")}
    assert "data-exhaustion" in slugs


# ---- borrowed from Utopia (docs/decisions/0001) --------------------------

from mindpalace.models import ClaimAssertion, Drop  # noqa: E402


def test_sync_projects_drops_untyped_and_validity_into_the_cache(conn, vault, config):
    _, store = vault
    note = Note(
        id="n_10",
        derived_from="c_10",
        created="2026-09-01T00:00:00Z",
        author="llm",
        body="Body.",
        entities=(EntityInstance("acme", None, "A company.", proposed_type="organisation"),),
        relationship_assertions=(
            RelationshipAssertion(
                "x_10", "star-wars", "geforce-now", None, 5, "playable", proposed_type="available on"
            ),
            RelationshipAssertion(
                "x_11", "a", "b", "contradicts", 5, "d", direction_corrected=True
            ),
        ),
        claim_assertions=(
            ClaimAssertion("k_10", "acme", "HQ moved.", valid_from="2026-03-15", valid_to="unknown"),
        ),
        drops=(Drop("relationship", "missing_field", "target", "a -> ?"),),
    )
    store.write_note(note, "acme")
    sync(conn, store, config, StubEmbedder(), {})

    row = conn.execute("SELECT type, proposed_type FROM assertions WHERE id = 'x_10'").fetchone()
    assert row["type"] is None and row["proposed_type"] == "available on"
    assert conn.execute("SELECT direction_corrected FROM assertions WHERE id = 'x_11'").fetchone()[0] == 1
    claim = conn.execute("SELECT valid_from, valid_to FROM claims WHERE id = 'k_10'").fetchone()
    assert (claim["valid_from"], claim["valid_to"]) == ("2026-03-15", "unknown")
    drop = conn.execute("SELECT note_id, kind, reason, detail, example FROM drops").fetchone()
    assert tuple(drop) == ("n_10", "relationship", "missing_field", "target", "a -> ?")
    proposals = {
        (r["kind"], r["proposed"]): (r["count"], r["example"])
        for r in conn.execute("SELECT kind, proposed, count, example FROM vocabulary_proposals")
    }
    assert proposals[("edge", "available-on")][0] == 1
    assert "star-wars" in proposals[("edge", "available-on")][1]
    assert proposals[("entity", "organisation")][0] == 1


def test_sync_applies_merges_and_adoptions_and_tracks_their_logs_for_drift(conn, vault, config):
    paths, store = vault
    add_note(store, "n_20", "open-ai")
    add_note(store, "n_21", "openai")
    sync(conn, store, config, StubEmbedder(), {}, merges={"open-ai": "openai"})
    slugs = {r["slug"] for r in conn.execute("SELECT slug FROM entities")}
    assert slugs == {"openai"}
    assert not has_drift(conn, store)
    paths.merges_log.parent.mkdir(exist_ok=True)
    paths.merges_log.write_text('{"x": 1}\n')
    assert has_drift(conn, store)
    sync(conn, store, config, StubEmbedder(), {})
    paths.vocabulary_log.write_text('{"x": 1}\n')
    assert has_drift(conn, store)


def test_a_merge_cycle_is_quarantined_as_an_issue(conn, vault, config):
    _, store = vault
    add_note(store, "n_30", "alpha")
    report = sync(conn, store, config, StubEmbedder(), {}, merges={"alpha": "beta", "beta": "alpha"})
    kinds = {issue[1] for issue in report.issues}
    assert "merge_cycle" in kinds
    assert {r["slug"] for r in conn.execute("SELECT slug FROM entities")} >= {"alpha"}


class _ProfileEmbedder:
    """Two entity profiles that mention SAME embed identically; all else random."""

    model_id = "profile-test"
    dim = 8

    def embed(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            if "SAME" in text:
                rows.append(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            else:
                rows.append(StubEmbedder(dim=8).embed([text])[0])
        return np.vstack(rows)


def test_similar_entities_by_embedding_are_reported_once_and_kept_pairs_are_not(conn, vault, config):
    _, store = vault
    for note_id, slug in (("n_40", "large-language-model"), ("n_41", "llm-large-model")):
        store.write_note(
            Note(
                id=note_id, derived_from=f"c_{note_id[2:]}", created="2026-08-01T00:00:00Z",
                author="llm", body="Body.",
                entities=(EntityInstance(slug, "concept", "SAME thing."),),
            ),
            slug,
        )
    add_note(store, "n_42", "sourdough")
    report = sync(conn, store, config, _ProfileEmbedder(), {})
    similar = [issue for issue in report.issues if issue[1] == "similar_entity"]
    assert len(similar) == 1
    assert "large-language-model" in similar[0][2] and "llm-large-model" in similar[0][2]
    assert "merge_entities" in similar[0][2]
    report = sync(
        conn, store, config, _ProfileEmbedder(), {},
        kept={("large-language-model", "llm-large-model")},
    )
    assert not [issue for issue in report.issues if issue[1] == "similar_entity"]


def test_a_stale_cache_schema_is_rebuilt_on_connect(tmp_path):
    import sqlite3

    path = tmp_path / ".graph" / "mindpalace.db"
    connection = db.connect(path)
    db.create_schema(connection)
    connection.execute("PRAGMA user_version = 0")
    connection.execute("DROP TABLE drops")
    connection.commit()
    connection.close()
    connection = db.connect(path)
    db.create_schema(connection)
    assert connection.execute("SELECT COUNT(*) FROM drops").fetchone()[0] == 0
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_a_schema_upgrade_keeps_the_communities_table(tmp_path):
    path = tmp_path / ".graph" / "mindpalace.db"
    connection = db.connect(path)
    db.create_schema(connection)
    connection.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES ('g_1', 0, NULL, 'a,b')"
    )
    connection.execute("PRAGMA user_version = 0")
    connection.commit()
    connection.close()
    connection = db.connect(path)
    db.create_schema(connection)
    assert connection.execute("SELECT COUNT(*) FROM communities").fetchone()[0] == 1


def test_a_schema_text_change_forces_a_rebuild_even_at_the_same_hand_version(tmp_path, monkeypatch):
    """The failure that hit the real vault: a column added to SCHEMA after the
    version was stamped. The version must follow the text, not a hand count."""
    path = tmp_path / ".graph" / "mindpalace.db"
    connection = db.connect(path)
    db.create_schema(connection)
    connection.execute("ALTER TABLE entities DROP COLUMN merged_from")
    connection.commit()
    connection.close()
    # Same code, same version: the stale shape survives, as it did in the wild.
    connection = db.connect(path)
    db.create_schema(connection)
    assert "merged_from" not in {r[1] for r in connection.execute("PRAGMA table_info(entities)")}
    connection.close()
    # Any change to the schema text changes the version and rebuilds.
    monkeypatch.setattr(db, "SCHEMA", db.SCHEMA + "\n-- shape moved\n")
    monkeypatch.setattr(db, "SCHEMA_VERSION", db.SCHEMA_VERSION + 1)
    connection = db.connect(path)
    db.create_schema(connection)
    assert "merged_from" in {r[1] for r in connection.execute("PRAGMA table_info(entities)")}


def test_multi_unit_captures_are_indexed_by_unit_not_by_body(conn, vault, config):
    from datetime import UTC, datetime

    _, store = vault
    body = "## Page 1\n\nalpha text here\n\n## Page 2\n\nbeta text here"
    a_end = body.index("## Page 2")
    store.write_capture(
        Capture("c_01", "2026-09-04T00:00:00Z", "file", None, body, units=((0, a_end), (a_end, len(body)))),
        datetime(2026, 9, 4, tzinfo=UTC),
    )
    store.write_capture(
        Capture("c_02", "2026-09-04T00:01:00Z", "manual", None, "single"),
        datetime(2026, 9, 4, 0, 1, tzinfo=UTC),
    )
    sync(conn, store, config, StubEmbedder(), {})
    units = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM text_units")}
    assert set(units) == {"u_01_0000", "u_01_0001", "u_02_0000"}
    assert units["u_01_0001"]["locator"] == "Page 2" and units["u_01_0001"]["text"].startswith("## Page 2")
    assert units["u_01_0000"]["locator"] == "Page 1"
    docs = {r["doc_id"]: r["kind"] for r in conn.execute("SELECT doc_id, kind FROM docs")}
    assert docs["u_01_0000"] == "unit" and docs["u_01_0001"] == "unit"
    assert "c_01" not in docs and docs["c_02"] == "capture" and "u_02_0000" not in docs


def test_provenance_is_projected_into_the_cache(conn, vault, config):
    _, store = vault
    store.write_note(
        Note(
            id="n_50", derived_from="c_50", created="2026-09-04T00:00:00Z", author="llm", body="B.",
            entities=(EntityInstance("a", "concept", "d", text_unit_ids=("u_50_0000",)),),
            relationship_assertions=(
                RelationshipAssertion("x_50", "a", "b", "contradicts", 5, "d", text_unit_ids=("u_50_0001",)),
            ),
        ),
        "a",
    )
    sync(conn, store, config, StubEmbedder(), {})
    rows = {(r["item_id"], r["unit_id"]) for r in conn.execute("SELECT item_id, unit_id FROM provenance")}
    assert rows == {("e_a", "u_50_0000"), ("e_a", "u_50_0001"), ("e_b", "u_50_0001"), ("x_50", "u_50_0001")}


def test_a_duplicate_capture_id_is_quarantined_not_fatal(conn, vault, config):
    from datetime import UTC, datetime

    paths, store = vault
    written = store.write_capture(
        Capture("c_dup", "2026-09-04T00:00:00Z", "manual", None, "first"),
        datetime(2026, 9, 4, tzinfo=UTC),
    )
    (paths.captures / (written.stem + " (copy).md")).write_text(written.read_text())
    report = sync(conn, store, config, StubEmbedder(), {})
    assert conn.execute("SELECT COUNT(*) FROM text_units WHERE capture_id = 'c_dup'").fetchone()[0] == 1
    kinds = {issue[1] for issue in report.issues}
    assert "duplicate_capture_id" in kinds


def test_a_unit_straddling_a_page_boundary_keeps_the_page_it_starts_on(conn, vault, config):
    from datetime import UTC, datetime

    _, store = vault
    body = "## Page 1\n\nstart on one\n\n## Page 2\n\nend on two"
    cut = body.index("start")
    store.write_capture(
        Capture("c_s", "2026-09-04T00:00:00Z", "file", None, body, units=((0, cut), (cut, len(body)))),
        datetime(2026, 9, 4, tzinfo=UTC),
    )
    sync(conn, store, config, StubEmbedder(), {})
    locators = {r["ordinal"]: r["locator"] for r in conn.execute("SELECT ordinal, locator FROM text_units WHERE capture_id = 'c_s'")}
    assert locators == {0: "Page 1", 1: "Page 1"}
