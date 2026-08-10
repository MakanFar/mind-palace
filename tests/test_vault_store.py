from datetime import UTC, datetime

import pytest

from mindpalace.atomic import ConflictError
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
