from datetime import UTC, datetime

from mindpalace.vault.paths import VaultPaths


def test_capture_filename_includes_ulid_suffix(tmp_path):
    paths = VaultPaths(tmp_path)
    created = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    path = paths.capture_path("c_01HQABCDEF", created)
    assert path.parent == tmp_path / "captures"
    assert path.name == "2026-08-08-1422-ABCDEF.md"


def test_two_captures_in_the_same_minute_do_not_collide(tmp_path):
    """Without the ULID suffix one capture would silently overwrite the other."""
    paths = VaultPaths(tmp_path)
    created = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    first = paths.capture_path("c_01HQAAAAAA", created)
    second = paths.capture_path("c_01HQBBBBBB", created)
    assert first != second


def test_note_path_combines_id_and_slug(tmp_path):
    paths = VaultPaths(tmp_path)
    path = paths.note_path("n_01hq", "Scaling Plateau")
    assert path == tmp_path / "notes" / "n_01hq-scaling-plateau.md"


def test_entity_and_community_paths(tmp_path):
    paths = VaultPaths(tmp_path)
    assert paths.entity_path("scaling-laws") == tmp_path / "entities" / "scaling-laws.md"
    assert paths.community_path("g_01ab") == tmp_path / "communities" / "g_01ab.md"


def test_community_path_is_stable_across_retitling(tmp_path):
    """Titles change; lineage ids do not. A title in the filename would leave a
    second file behind on every retitle, and read_report would glob arbitrarily."""
    paths = VaultPaths(tmp_path)
    assert paths.community_path("g_01ab") == paths.community_path("g_01ab")


def test_well_known_locations(tmp_path):
    paths = VaultPaths(tmp_path)
    assert paths.mindpalace_md == tmp_path / "MINDPALACE.md"
    assert paths.graph_db == tmp_path / ".graph" / "mindpalace.db"
    assert paths.decisions_log == tmp_path / ".mindpalace" / "decisions.jsonl"
    assert paths.op_log == tmp_path / ".mindpalace" / "log.jsonl"
    assert paths.lock == tmp_path / ".mindpalace" / "lock"
