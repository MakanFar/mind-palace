import os

import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session, VaultLockedError


def open_session(root):
    return Session(root, init=True, embedder=StubEmbedder())


def test_session_initialises_a_fresh_vault(tmp_path):
    with open_session(tmp_path) as session:
        assert session.paths.mindpalace_md.exists()
        assert session.config.schema_version == 1


# ---- concurrency: exclusion belongs around writes, not around the process ---


def test_two_sessions_can_hold_the_same_vault_open(tmp_path):
    """The lock used to be an exclusive `flock` taken for the whole session
    lifetime, so the *second* server on a vault exited 2 with `vault ... is
    already open (pid N)`. Every MCP client that starts one stdio server per
    editor session (Claude Code does) therefore had exactly one working
    session and a queue of dead ones behind it. Opening is a read; only
    writes need to exclude each other."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        assert first.opened is True
        assert second.opened is True


def test_a_write_excludes_a_writer_in_another_session(tmp_path):
    """The single-writer invariant survives the change above: one flock, taken
    for the duration of a write rather than the duration of the process."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        second.lock_timeout = 0.05
        with first.write_lock():
            with pytest.raises(VaultLockedError, match="already open"):
                with second.write_lock():
                    pass


def test_a_finished_write_releases_the_lock(tmp_path):
    """The mirror of the test above: exclusion that outlives the write is the
    process-lifetime lock again, wearing a different name."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        second.lock_timeout = 0.05
        with first.write_lock():
            pass
        with second.write_lock():
            assert second.opened is True


def test_an_operation_excludes_a_writer_in_another_session(tmp_path):
    """`operation()` is the framing every mutating tool goes through, so it is
    what must take the lock; a `write_lock()` nobody calls protects nothing."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        second.lock_timeout = 0.05
        with first.operation({"tool": "test"}):
            with pytest.raises(VaultLockedError, match="already open"):
                with second.operation({"tool": "test"}):
                    pass


def test_a_write_refused_for_the_lock_leaves_no_pending_operation(tmp_path):
    """The lock is taken before the op log is written. Taking it after would
    make every contended write look like a crashed one, and the next open
    would replay it."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        second.lock_timeout = 0.05
        with first.write_lock():
            with pytest.raises(VaultLockedError):
                with second.operation({"tool": "test"}):
                    pass
        assert second.oplog.pending() == []


def test_nested_write_locks_stay_held_until_the_outermost_exits(tmp_path):
    """`resync()` locks and is itself called from inside `operation()` (see
    tools.py). A non-reentrant lock would have the inner exit unlock the
    vault while the outer write is still running."""
    with open_session(tmp_path) as first, open_session(tmp_path) as second:
        second.lock_timeout = 0.05
        with first.write_lock():
            with first.write_lock():
                pass
            with pytest.raises(VaultLockedError, match="already open"):
                with second.write_lock():
                    pass


def test_a_write_in_one_session_is_visible_in_another(tmp_path):
    """Two live sessions share Tier 1 files and the one Tier 3 cache file, so
    a capture saved by one is readable by the other with no reopen."""
    from mindpalace.tools import read, save_capture

    with open_session(tmp_path) as writer, open_session(tmp_path) as reader:
        capture = save_capture(writer, "Something worth remembering.")
        assert read(reader, capture["id"])["id"] == capture["id"]


def test_closing_a_session_leaves_another_sessions_lock_armed(tmp_path):
    """`close()` used to `unlink` the lock file. A session still holding a
    flock on the now-unlinked inode does not exclude a newcomer, which creates
    a fresh file at the same path and locks that instead: two writers, each
    believing it is exclusive."""
    holder = open_session(tmp_path).open()
    closer = open_session(tmp_path).open()
    late = open_session(tmp_path).open()
    late.lock_timeout = 0.05
    try:
        with holder.write_lock():
            closer.close()
            with pytest.raises(VaultLockedError, match="already open"):
                with late.write_lock():
                    pass
    finally:
        late.close()
        holder.close()


def test_a_vault_with_nothing_to_heal_opens_while_another_session_writes(tmp_path):
    """`heal()` runs on every open. Taking the write lock before checking
    whether there is anything to heal puts every new session behind whatever
    the other one is doing -- and a `rebuild` outlasting the timeout would
    fail the open outright, which is the startup failure this whole change
    exists to remove."""
    with open_session(tmp_path) as writer:
        with writer.write_lock():
            with Session(
                tmp_path, embedder=StubEmbedder(), lock_timeout=0.05
            ) as opened:
                assert opened.opened is True
                assert opened.heal() is False


def test_a_lock_error_that_is_not_contention_is_not_reported_as_contention(tmp_path):
    """Only `EWOULDBLOCK` means "someone else is writing". Retrying anything
    else until the timeout spends 60s to raise a `VaultLockedError` naming an
    innocent pid, hiding the real fault (a closed descriptor, say)."""
    import errno
    import fcntl

    with open_session(tmp_path) as session:
        def refuse(_fd, _op):
            raise OSError(errno.EBADF, "bad file descriptor")

        original = fcntl.flock
        fcntl.flock = refuse
        try:
            with pytest.raises(OSError) as caught:
                with session.write_lock():
                    pass
        finally:
            fcntl.flock = original

    assert caught.value.errno == errno.EBADF
    assert not isinstance(caught.value, VaultLockedError)


def test_a_stale_pid_in_the_lock_file_does_not_block_a_write(tmp_path):
    """flock is released by the kernel when the holder dies, so the recorded
    pid is diagnostics only and a leftover one never blocks a write."""
    with open_session(tmp_path) as session:
        lock_path = session.paths.lock
    lock_path.write_text("999999")  # pid that cannot be running
    with open_session(tmp_path) as session:
        with session.write_lock():
            assert lock_path.read_text().strip() == str(os.getpid())


def test_changing_the_embedder_rebuilds_the_cache(tmp_path):
    """A swapped model does not error — vectors filter on model_id, so it would
    silently return nothing. Startup must detect and rebuild."""
    with open_session(tmp_path) as session:
        session.conn.execute(
            "INSERT INTO docs (doc_id, kind, title, text) VALUES "
            "('n_stale', 'note', 't', 'text')"
        )
        session.conn.commit()

    with Session(tmp_path, embedder=StubEmbedder(dim=32)) as session:
        from mindpalace.index import db

        assert db.read_meta(session.conn) == ("stub-32", 32)
        assert session.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE doc_id = 'n_stale'"
        ).fetchone()[0] == 0


def test_an_external_config_edit_is_reloaded(tmp_path):
    with open_session(tmp_path) as session:
        original = session.config.thresholds.abstain_cosine_floor
        session.paths.mindpalace_md.write_text(
            session.paths.mindpalace_md.read_text().replace(
                "abstain_cosine_floor: 0.35", "abstain_cosine_floor: 0.6"
            )
        )
        assert session.heal() is True
        assert session.config.thresholds.abstain_cosine_floor == 0.6
        assert original == 0.35


def test_operation_commits_on_success(tmp_path):
    with open_session(tmp_path) as session:
        with session.operation({"tool": "test"}):
            pass
        assert session.oplog.pending() == []


def test_operation_leaves_a_pending_record_on_failure(tmp_path):
    with open_session(tmp_path) as session:
        with pytest.raises(RuntimeError):
            with session.operation({"tool": "test"}):
                raise RuntimeError("boom")
        assert len(session.oplog.pending()) == 1


def test_heal_clears_pending_operations(tmp_path):
    with open_session(tmp_path) as session:
        with pytest.raises(RuntimeError):
            with session.operation({"tool": "test"}):
                raise RuntimeError("boom")

    with open_session(tmp_path) as session:
        assert session.oplog.pending() == []


def test_heal_reports_whether_it_did_work(tmp_path):
    with open_session(tmp_path) as session:
        assert session.heal() is False  # nothing pending, no drift


def break_the_embedder_config(tmp_path):
    """Leave the vault scaffolded but unopenable: `cloud` is a configured
    embedder kind that deliberately refuses to construct."""
    with open_session(tmp_path):
        pass

    md = session_paths_mindpalace_md(tmp_path)
    md.write_text(
        md.read_text().replace(
            "embedder: {kind: local, model: BAAI/bge-small-en-v1.5}",
            "embedder: {kind: cloud}",
        )
    )


def test_a_failed_open_leaves_the_vault_openable(tmp_path):
    """If a step after the lock file is opened raises (e.g. the configured
    embedder can't be constructed), the half-built session must not hold
    anything a later, good open would trip over."""
    from mindpalace.embed import EmbedderError

    break_the_embedder_config(tmp_path)

    with pytest.raises(EmbedderError):
        Session(tmp_path).open()  # no explicit embedder -> uses the broken config

    with open_session(tmp_path) as session:
        assert session.opened is True


def test_a_failed_open_does_not_leak_a_descriptor(tmp_path):
    """The lock file is opened before the steps that can fail, and a server
    that retries a bad open (or a long-lived process opening several vaults)
    would otherwise run out of descriptors."""
    from mindpalace.embed import EmbedderError

    break_the_embedder_config(tmp_path)

    def open_descriptor_count():
        return len(os.listdir("/dev/fd"))

    with pytest.raises(EmbedderError):
        Session(tmp_path).open()  # warm up: first failure may cache imports
    before = open_descriptor_count()

    for _ in range(5):
        with pytest.raises(EmbedderError):
            Session(tmp_path).open()

    assert open_descriptor_count() == before


def session_paths_mindpalace_md(root):
    from mindpalace.vault.paths import VaultPaths

    return VaultPaths(root).mindpalace_md


# ---- CRITICAL regression: a scalar `user:` value must not block open -----


def test_a_scalar_user_value_in_an_entity_page_does_not_block_open(tmp_path):
    """A regression introduced by this very fix wave's interaction between
    Important 8 and the unguarded `page.user.get(...)` call sites: adding
    `entities/` to the drift set means `Session.open()` -> `heal()` ->
    `resync()` now runs `sync()` over a hand-edited entity page during
    *open*, not only on the next write tool call. `sync`'s
    `_ambiguous_alias_issues` calls `page.user.get("aliases", [])`
    unguarded, so a scalar `user:` value -- exactly what the generated
    banner's own instruction ("put durable changes under `user:` in the
    front-matter") invites a human to type -- raised `AttributeError:
    'str' object has no attribute 'get'` straight out of `resync()`, and
    `main()`'s `except (ConfigError, VaultLockedError, EmbedderError)`
    doesn't cover it: the server would not start at all. Confirmed as a
    genuine delta between commits by the scoped re-review (vault opened,
    with writes broken, before this branch; refused to open at all after).

    Assert both halves: the vault still opens, and the assistant is told
    which file is wrong via review_queue -- not left to guess why startup
    silently produced an empty-looking vault.
    """
    with open_session(tmp_path) as session:
        entities_dir = session.paths.entities

    entities_dir.mkdir(parents=True, exist_ok=True)
    (entities_dir / "scalar-user.md").write_text(
        "---\n"
        "id: e_scalar-user\n"
        "type: concept\n"
        "generated_from: []\n"
        "input_hash: ''\n"
        "stale: false\n"
        "user: reviewed by me on tuesday\n"
        "---\n\n"
        "Some description a human wrote by hand.\n\n"
        "<!-- mindpalace:related -->\n"
        "<!-- /mindpalace:related -->\n"
    )

    from mindpalace.tools import review_queue

    with open_session(tmp_path) as session:
        assert session.opened is True

        issues = review_queue(session)["vault_issues"]
        assert any(
            issue["kind"] == "malformed_entity_page"
            and "scalar-user.md" in issue["path"]
            for issue in issues
        )


# ---- IMPORTANT 7: corrupt SQLite must be recovered, not raised ------------


def test_corrupt_sqlite_is_recovered_and_rebuilt_from_tier_1(tmp_path):
    """Spec §10: 'Corrupt SQLite -- detected on open, rebuilt from Tier 1.
    Nothing is lost.' A corrupted `.graph/mindpalace.db` used to raise
    `sqlite3.DatabaseError` straight out of `Session.open()`; `main`'s
    `except (ConfigError, VaultLockedError, EmbedderError)` does not cover
    it, so the user got a traceback and had to know to delete `.graph/` by
    hand. Tier 3 is delete-and-rebuild by construction, so recovery must be
    silent and complete: the note written before the corruption must still
    be there afterwards."""
    from mindpalace.tools import save_capture, write_note

    with open_session(tmp_path) as session:
        capture = save_capture(session, "Something worth remembering.")
        write_note(session, derived_from=capture["id"], content="Analysis text.")
        graph_db_path = session.paths.graph_db

    graph_db_path.write_bytes(b"not a sqlite database at all")

    with open_session(tmp_path) as session:
        assert session.opened is True
        note_count = session.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE kind = 'note'"
        ).fetchone()[0]
        assert note_count == 1


def test_corrupt_sqlite_does_not_leak_the_lock(tmp_path):
    """Recovery must not bypass the same lock-safety guarantee every other
    open-time failure has (see test_a_failed_open_does_not_leak_the_lock)."""
    with open_session(tmp_path) as session:
        graph_db_path = session.paths.graph_db

    graph_db_path.write_bytes(b"not a sqlite database at all")

    with open_session(tmp_path) as session:
        assert session.opened is True
