from mindpalace.index import db


def test_create_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    db.create_schema(conn)  # must not raise
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert db.TABLES <= names


def test_fts_table_supports_match(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
        ("n_01", "note", "Scaling", "the plateau is about data exhaustion"),
    )
    rows = list(
        conn.execute(
            "SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY rank",
            ("exhaustion",),
        )
    )
    assert [row[0] for row in rows] == ["n_01"]


def test_foreign_keys_and_wal_are_enabled(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_file_hashes_round_trip(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO files (path, hash) VALUES (?, ?)", ("notes/n_01.md", "sha256:aa")
    )
    conn.execute(
        "INSERT INTO files (path, hash) VALUES (?, ?) "
        "ON CONFLICT(path) DO UPDATE SET hash = excluded.hash",
        ("notes/n_01.md", "sha256:bb"),
    )
    assert conn.execute("SELECT hash FROM files").fetchone()[0] == "sha256:bb"


def test_cache_meta_is_absent_until_written(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    assert db.read_meta(conn) is None


def test_cache_meta_round_trips_and_upserts(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    db.write_meta(conn, "stub-64", 64)
    assert db.read_meta(conn) == ("stub-64", 64)
    db.write_meta(conn, "bge-small", 384)
    assert db.read_meta(conn) == ("bge-small", 384)
