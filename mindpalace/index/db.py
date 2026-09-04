"""Tier 3 SQLite cache. Deletable and fully rebuildable from Tier 1."""

from __future__ import annotations

import hashlib
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
        "drops",
        "vocabulary_proposals",
        "text_units",
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
    rank INTEGER NOT NULL DEFAULT 0,
    -- Comma-joined slugs merged into this one (docs/decisions/0001 §5).
    merged_from TEXT NOT NULL DEFAULT ''
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
    status TEXT NOT NULL,
    -- Untyped assertions (docs/decisions/0001 §1): type is NULL and the
    -- model's wording sits here.
    proposed_type TEXT,
    direction_corrected INTEGER NOT NULL DEFAULT 0
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
    status TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    supersedes TEXT
);

-- Projection of every note's `drops:` ledger (docs/decisions/0001 §2).
CREATE TABLE IF NOT EXISTS drops (
    note_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT NOT NULL,
    example TEXT
);

-- Derived from each capture's `units` offsets (docs/decisions/0002).
CREATE TABLE IF NOT EXISTS text_units (
    id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    locator TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_text_units_capture ON text_units(capture_id);

-- Out-of-vocabulary wordings, counted by normalised form (0001 §1).
CREATE TABLE IF NOT EXISTS vocabulary_proposals (
    kind TEXT NOT NULL,
    proposed TEXT NOT NULL,
    count INTEGER NOT NULL,
    example TEXT NOT NULL,
    ids TEXT NOT NULL,
    PRIMARY KEY (kind, proposed)
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


#: Derived from the schema text, not maintained by hand. The cache is
#: delete-and-rebuild by construction, so a mismatch is not migrated: the
#: tables are dropped and recreated, and the next sync repopulates them from
#: Tier 1. A hand-bumped number was tried first and failed in the obvious
#: way -- a column was added after the bump, a server built from that
#: in-between commit stamped the vault with the new number and the old shape,
#: and every later open failed on the first insert with "no column named".
#: Hashing the text means no one has to remember. Masked to 31 bits because
#: `PRAGMA user_version` is a signed 32-bit integer.
SCHEMA_VERSION = int(hashlib.sha256(SCHEMA.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the MCP runtime dispatches every synchronous
    # tool body through anyio.to_thread.run_sync (see
    # mcp/server/mcpserver/utilities/func_metadata.py), and `tools/call` is
    # not one of jsonrpc_dispatcher's inline_methods -- it is handled via
    # `task_group.start_soon`, so two pipelined `tools/call` requests can be
    # mid-flight on two different worker threads at once. A connection
    # created with the sqlite3 default would raise "SQLite objects created in
    # a thread can only be used in that same thread" on the very first real
    # tool call, since the connection is always created on whichever thread
    # opened the Session, never on a worker thread.
    #
    # This flag only stops that crash; it does not make concurrent use safe
    # by itself -- sqlite3.threadsafety is 3 here, so two threads sharing
    # this connection cannot corrupt its memory, but they *can* interleave
    # statements within its one shared implicit transaction (an unwrapped
    # multi-statement write racing a read, or one thread's transaction
    # committing another thread's in-flight write early). Mutual exclusion
    # across threads is `session.lock` (see Session.__init__), taken around
    # every tool dispatch in `server.py` -- the vault's `flock` only excludes
    # a second *process* and provides no help here.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the tables, first dropping every one of them if the file was
    written by a different schema text. `CREATE TABLE IF NOT EXISTS`
    alone would leave an old table missing the new columns and fail on the
    first insert; the version pragma is what lets an upgrade be a rebuild
    rather than a crash. `communities` is kept if its own columns still
    match, since `sync` never repopulates it (see below)."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current != SCHEMA_VERSION:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        # `communities` is written by `cluster`, not by `sync`, so dropping it
        # would orphan every existing report until someone reclusters -- and
        # reclustering mints new lineage ids. Keep it unless its own shape
        # moved.
        keep: set[str] = set()
        if "communities" in existing:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(communities)")
            }
            if columns == {"lineage_id", "level", "parent", "members"}:
                keep.add("communities")
        for table in sorted((TABLES - keep) & existing):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
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
