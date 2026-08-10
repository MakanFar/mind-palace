"""Tier 3 SQLite cache. Deletable and fully rebuildable from Tier 1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TABLES = frozenset(
    {
        "cache_meta",
        "files",
        "entities",
        "entity_sources",
        "assertions",
        "aggregates",
        "aggregate_members",
        "claims",
        "communities",
        "docs",
        "vectors",
        "vault_issues",
    }
)

SCHEMA = """
-- Single-row table describing what produced this cache. A mismatch against the
-- live embedder forces a rebuild (spec §5) instead of silently returning zero
-- vector results, which is what a changed model would otherwise cause.
CREATE TABLE IF NOT EXISTS cache_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    slug TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entity_sources (
    slug TEXT NOT NULL,
    note_id TEXT NOT NULL,
    PRIMARY KEY (slug, note_id)
);

CREATE TABLE IF NOT EXISTS assertions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    note_id TEXT NOT NULL,
    source TEXT,
    target TEXT,
    type TEXT,
    strength INTEGER,
    subject TEXT,
    text TEXT,
    description TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aggregates (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    target TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    mean_strength REAL NOT NULL DEFAULT 0,
    traversable INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS aggregate_members (
    key TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    PRIMARY KEY (key, assertion_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS communities (
    lineage_id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    parent TEXT,
    members TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    doc_id UNINDEXED,
    kind UNINDEXED,
    title,
    text
);

CREATE TABLE IF NOT EXISTS vectors (
    doc_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_issues (
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aggregates_source ON aggregates(source);
CREATE INDEX IF NOT EXISTS idx_aggregates_target ON aggregates(target);
CREATE INDEX IF NOT EXISTS idx_assertions_note ON assertions(note_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the MCP server dispatches every synchronous tool
    # body through anyio.to_thread.run_sync, so a call executes on a worker
    # thread distinct from whichever thread opened the Session (see
    # mindpalace/server.py). A connection created with the sqlite3 default would
    # raise "SQLite objects created in a thread can only be used in that same
    # thread" on the very first real tool call. The vault's own concurrency
    # story (Session.py's flock, one open session per process, stdio transport
    # processing one request at a time) already guarantees this connection is
    # never touched by two threads at once -- only ever a *different* thread
    # each call, sequentially -- so disabling sqlite3's same-thread check
    # (rather than adding a lock) is a correct fit, not just a workaround.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def read_meta(conn: sqlite3.Connection) -> tuple[str, int] | None:
    row = conn.execute("SELECT model_id, dim FROM cache_meta WHERE id = 1").fetchone()
    return None if row is None else (row["model_id"], row["dim"])


def write_meta(conn: sqlite3.Connection, model_id: str, dim: int) -> None:
    conn.execute(
        "INSERT INTO cache_meta (id, model_id, dim) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id, "
        "dim = excluded.dim",
        (model_id, dim),
    )
