import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session, VaultLockedError


def open_session(root):
    return Session(root, init=True, embedder=StubEmbedder())


def test_session_initialises_a_fresh_vault(tmp_path):
    with open_session(tmp_path) as session:
        assert session.paths.mindpalace_md.exists()
        assert session.config.schema_version == 1


def test_lock_prevents_a_second_writer(tmp_path):
    with open_session(tmp_path):
        with pytest.raises(VaultLockedError, match="already open"):
            open_session(tmp_path).open()


def test_lock_is_released_on_close(tmp_path):
    with open_session(tmp_path):
        pass
    with open_session(tmp_path) as session:
        assert session.paths.lock.exists()


def test_stale_lock_from_a_dead_process_is_reclaimed(tmp_path):
    """flock is released by the kernel when the holder dies, so a leftover pid
    file never blocks a restart."""
    with open_session(tmp_path) as session:
        lock_path = session.paths.lock
    lock_path.write_text("999999")  # pid that cannot be running
    with open_session(tmp_path) as session:
        assert session.paths.lock.read_text() != "999999"


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


def test_a_failed_open_does_not_leak_the_lock(tmp_path):
    """If a step after the lock is taken raises (e.g. the configured embedder
    can't be constructed), the lock must be released -- otherwise a vault that
    failed to open once can never be opened again in the same process."""
    from mindpalace.embed import EmbedderError

    with open_session(tmp_path):
        pass  # scaffold the vault

    text = session_paths_mindpalace_md(tmp_path).read_text()
    session_paths_mindpalace_md(tmp_path).write_text(
        text.replace("embedder: {kind: local, model: BAAI/bge-small-en-v1.5}",
                     "embedder: {kind: cloud}")
    )

    with pytest.raises(EmbedderError):
        Session(tmp_path).open()  # no explicit embedder -> uses the broken config

    # The lock must have been released despite the failure above.
    with open_session(tmp_path) as session:
        assert session.opened is True


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
