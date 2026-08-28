from datetime import UTC, datetime

import pytest

from mindpalace.atomic import ConflictError
from mindpalace.frontmatter import FrontMatterError
from mindpalace.models import Capture, CommunityReport, EntityPage, Note
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


def make_capture(capture_id="c_01HQABCDEF"):
    return Capture(
        id=capture_id,
        created="2026-08-08T14:22:00Z",
        source="manual",
        why=None,
        text="The plateau talk is about data exhaustion.",
    )


def test_capture_write_then_read(store):
    capture = make_capture()
    path = store.write_capture(capture, datetime(2026, 8, 8, 14, 22, tzinfo=UTC))
    assert store.read_capture(path) == capture


def test_capture_write_refuses_to_overwrite(store):
    capture = make_capture()
    when = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    store.write_capture(capture, when)
    with pytest.raises(ConflictError):
        store.write_capture(capture, when)


def test_note_write_then_read_and_iterate(store):
    note = Note(
        id="n_01hq",
        derived_from="c_01hq",
        created="2026-08-08T14:25:00Z",
        author="llm",
        body="Analysis.",
    )
    store.write_note(note, "scaling plateau")
    assert list(store.iter_notes()) == [note]


def test_entity_page_round_trips_with_user_block_and_related(store):
    page = EntityPage(
        slug="scaling-laws",
        type="concept",
        description="Compute, data, and loss.",
        generated_from=["n_01hq"],
        input_hash="sha256:abc",
        stale=False,
        user={"aliases": ["scaling law"]},
        related=["contradicts [[data-exhaustion]]"],
    )
    store.write_entity_page(page)
    assert store.read_entity_page("scaling-laws") == page


def test_read_missing_entity_page_returns_none(store):
    assert store.read_entity_page("nobody") is None


def test_report_round_trips(store):
    report = CommunityReport(
        lineage_id="g_01ab",
        level=0,
        title="Scaling Debate",
        summary="A cluster about scaling limits.",
        rank=6.5,
        findings=[{"summary": "Data supply dominates", "explanation": "…"}],
        cites=["e_scaling-laws", "x_01ab"],
        generated_from=["e_scaling-laws"],
        input_hash="sha256:def",
        stale=False,
    )
    store.write_report(report)
    assert store.read_report("g_01ab") == report


def test_retitling_a_report_does_not_leave_a_second_file(store):
    for title in ("First Title", "Second Title"):
        store.write_report(
            CommunityReport(
                lineage_id="g_01ab",
                level=0,
                title=title,
                summary="Same community, renamed.",
                rank=5.0,
            )
        )
    assert len(list(store.paths.communities.glob("*.md"))) == 1
    assert store.read_report("g_01ab").title == "Second Title"


def test_entity_page_with_empty_related_round_trips(store):
    """An entity page with no related links should round-trip correctly."""
    page = EntityPage(
        slug="no-links",
        type="concept",
        description="A standalone concept.",
        generated_from=["n_01hq"],
        input_hash="sha256:abc",
        stale=False,
        user={},
        related=[],
    )
    store.write_entity_page(page)
    read_back = store.read_entity_page("no-links")
    assert read_back == page
    assert read_back.related == []


def test_entity_page_description_can_quote_marker_text(store):
    """Marker text in the middle of a line should not be treated as a boundary.

    Regression test: the old partition()-based implementation would incorrectly
    split on the first occurrence of the marker text anywhere, even mid-line.
    """
    page = EntityPage(
        slug="test-page",
        type="concept",
        description='The doc mentions "<!-- mindpalace:related -->" but this is just text.',
        generated_from=["n_01hq"],
        input_hash="sha256:abc",
        stale=False,
        user={},
        related=["actual-related-link"],
    )
    store.write_entity_page(page)
    read_back = store.read_entity_page("test-page")
    assert read_back == page
    assert read_back.description == page.description
    assert read_back.related == page.related


def test_write_entity_page_rejects_marker_as_standalone_line(store):
    """A description containing a marker as a complete line should raise ValueError."""
    page = EntityPage(
        slug="bad-page",
        type="concept",
        description="Some text\n<!-- mindpalace:related -->\nMore text",
        generated_from=["n_01hq"],
        input_hash="sha256:abc",
        stale=False,
    )
    with pytest.raises(ValueError, match="mindpalace:related"):
        store.write_entity_page(page)


# ---- IMPORTANT 3: content below the related block must survive ------------


def test_a_hand_written_section_below_the_related_block_round_trips(store):
    """`_split_related` used to return only `lines[:open_idx]` as the
    description and drop everything after the close marker -- so a section
    a human appended below the machine-owned block vanished the next time
    `write_entity_page` rebuilt the file (which every write tool's rebuild
    call does)."""
    page = EntityPage(
        slug="scaling-laws",
        type="concept",
        description="Compute, data, and loss.",
        related=["contradicts [[data-exhaustion]]"],
    )
    store.write_entity_page(page)
    path = store.paths.entity_path("scaling-laws")
    path.write_text(
        path.read_text() + "\n## My own notes\n\nHand written, must survive.\n"
    )

    read_back = store.read_entity_page("scaling-laws")
    assert "Hand written, must survive." in read_back.trailing


def test_regenerating_an_entity_page_preserves_the_trailing_section(store):
    """The regression itself: read a page with a hand-written trailing
    section, change something machine-owned, write it back -- the trailing
    section must still be there afterwards."""
    store.write_entity_page(
        EntityPage(slug="scaling-laws", type="concept", description="Original.")
    )
    path = store.paths.entity_path("scaling-laws")
    path.write_text(
        path.read_text() + "\n## My own notes\n\nHand written, must survive.\n"
    )

    page = store.read_entity_page("scaling-laws")
    page.description = "Updated by rebuild."
    store.write_entity_page(page)

    survivor = store.read_entity_page("scaling-laws")
    assert "Hand written, must survive." in survivor.trailing
    assert survivor.description == "Updated by rebuild."


# ---- CRITICAL 2: malformed Tier-2 files must degrade, not raise bare ------


def test_read_entity_page_raises_frontmattererror_on_bad_yaml(store):
    path = store.paths.entity_path("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nnot: [closed\n")
    with pytest.raises(FrontMatterError):
        store.read_entity_page("broken")


def test_read_entity_page_raises_frontmattererror_on_valid_yaml_missing_type(store):
    """Reproduces the finding's exact example: dropping a page containing
    only `---\\ntitle: oops\\n---` used to die with a bare `KeyError:
    'type'`, not an actionable, catchable error."""
    path = store.paths.entity_path("hand-made")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: oops\n---\n")
    with pytest.raises(FrontMatterError, match="type"):
        store.read_entity_page("hand-made")


def test_read_entity_page_raises_frontmattererror_on_a_scalar_user_value(store):
    """CRITICAL regression from the fix wave itself: the generated banner's
    own instruction -- "put durable changes under `user:` in the
    front-matter" -- invites exactly this. `user: reviewed by me on
    tuesday` is valid YAML (a scalar, not a mapping), so it round-tripped
    past `parse()` and into `EntityPage.user` untouched; every consumer
    (`_ambiguous_alias_issues`, `_resolve_slug`, `_known_entities`,
    `rebuild`'s `existing.user`) calls `.get(...)` on it unguarded and
    raised `AttributeError: 'str' object has no attribute 'get'`. Adding
    `entities/` to the drift set (Important 8) means `Session.open()` ->
    `heal()` -> `resync()` now hits this on open, not just on the next
    write -- so this used to make the vault refuse to open at all."""
    path = store.paths.entity_path("scalar-user")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "id: e_scalar-user\n"
        "type: concept\n"
        "user: reviewed by me on tuesday\n"
        "---\n\nSome description.\n"
    )
    with pytest.raises(FrontMatterError, match="mapping"):
        store.read_entity_page("scalar-user")


def test_iter_entity_pages_skips_a_page_with_a_scalar_user_value(store):
    store.write_entity_page(
        EntityPage(slug="good", type="concept", description="Fine.")
    )
    bad_path = store.paths.entity_path("bad")
    bad_path.write_text(
        "---\nid: e_bad\ntype: concept\nuser: reviewed by me on tuesday\n---\n\nBad.\n"
    )

    pages = list(store.iter_entity_pages())
    assert [page.slug for page in pages] == ["good"]


def test_iter_entity_pages_skips_a_malformed_page_instead_of_raising(store):
    store.write_entity_page(
        EntityPage(slug="good", type="concept", description="Fine.")
    )
    bad_path = store.paths.entity_path("bad")
    bad_path.write_text("---\ntitle: oops\n---\n")

    pages = list(store.iter_entity_pages())
    assert [page.slug for page in pages] == ["good"]


def test_read_report_raises_frontmattererror_on_missing_required_key(store):
    path = store.paths.community_path("g_bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: oops\n---\n")
    with pytest.raises(FrontMatterError):
        store.read_report("g_bad")


def test_iter_reports_skips_a_malformed_report_instead_of_raising(store):
    store.write_report(
        CommunityReport(lineage_id="g_good", level=0, title="T", summary="S", rank=1.0)
    )
    bad_path = store.paths.community_path("g_bad")
    bad_path.write_text("---\ntitle: oops\n---\n")

    reports = list(store.iter_reports())
    assert [report.lineage_id for report in reports] == ["g_good"]


# ---- IMPORTANT 5: Tier-2 writes must CAS-check like Tier-1 writes do ------


def test_write_entity_page_detects_a_concurrent_edit(store):
    """write_entity_page used plain atomic_write after a read-modify-write:
    a concurrent Obsidian save between the read and the write was silently
    clobbered instead of surfaced as a conflict (spec §9.2)."""
    store.write_entity_page(
        EntityPage(slug="scaling-laws", type="concept", description="Original.")
    )
    read_for_edit = store.read_entity_page("scaling-laws")

    concurrent = store.read_entity_page("scaling-laws")
    concurrent.description = "Edited concurrently in Obsidian."
    store.write_entity_page(concurrent)

    read_for_edit.description = "My conflicting edit."
    with pytest.raises(ConflictError):
        store.write_entity_page(read_for_edit)


def test_write_entity_page_without_a_prior_read_is_unconditional(store):
    """A page built fresh (not obtained via `read_entity_page`) was never
    part of a read-then-write sequence, so it must keep working
    unconditionally -- this is how every test (and `rebuild`'s
    create-a-missing-page path) seeds or replaces a page today."""
    store.write_entity_page(
        EntityPage(slug="scaling-laws", type="concept", description="First.")
    )
    store.write_entity_page(
        EntityPage(slug="scaling-laws", type="concept", description="Second.")
    )
    assert store.read_entity_page("scaling-laws").description == "Second."


def test_write_report_detects_a_concurrent_edit(store):
    store.write_report(
        CommunityReport(lineage_id="g_01", level=0, title="T", summary="S", rank=5.0)
    )
    read_for_edit = store.read_report("g_01")

    concurrent = store.read_report("g_01")
    concurrent.summary = "Edited concurrently."
    store.write_report(concurrent)

    read_for_edit.summary = "My conflicting edit."
    with pytest.raises(ConflictError):
        store.write_report(read_for_edit)
