import pytest

from mindpalace.atomic import ConflictError, atomic_write, cas_write, content_hash


def test_content_hash_is_stable_and_prefixed():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello").startswith("sha256:")
    assert content_hash("hello") != content_hash("world")


def test_atomic_write_creates_file_and_parents(tmp_path):
    target = tmp_path / "nested" / "note.md"
    atomic_write(target, "body\n")
    assert target.read_text() == "body\n"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "body\n")
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_cas_write_succeeds_when_hash_matches(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "original\n")
    cas_write(target, "updated\n", content_hash("original\n"))
    assert target.read_text() == "updated\n"


def test_cas_write_aborts_when_file_changed_underneath(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "original\n")
    stale = content_hash("original\n")
    atomic_write(target, "edited in obsidian\n")  # external edit

    with pytest.raises(ConflictError, match="changed on disk"):
        cas_write(target, "updated\n", stale)
    assert target.read_text() == "edited in obsidian\n"


def test_cas_write_with_none_expects_absent_file(tmp_path):
    target = tmp_path / "new.md"
    cas_write(target, "fresh\n", None)
    assert target.read_text() == "fresh\n"


def test_cas_write_with_none_rejects_existing_file(tmp_path):
    target = tmp_path / "new.md"
    atomic_write(target, "already here\n")
    with pytest.raises(ConflictError, match="already exists"):
        cas_write(target, "fresh\n", None)
