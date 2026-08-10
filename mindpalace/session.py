"""Process-level vault session: locking, crash healing, operation framing."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from mindpalace.config import Config, load_config, open_vault
from mindpalace.embed import Embedder, get_embedder
from mindpalace.index import db
from mindpalace.index.sync import has_drift, sync
from mindpalace.oplog import DecisionLog, OpLog
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


class VaultLockedError(RuntimeError):
    """Raised when another server already holds this vault."""


class Session:
    def __init__(
        self, root: Path, *, init: bool = False, embedder: Embedder | None = None
    ) -> None:
        self._root = Path(root)
        self._init = init
        self._explicit_embedder = embedder
        self._lock_fd: int | None = None
        self.paths: VaultPaths
        self.config: Config
        self.store: VaultStore
        self.embedder: Embedder
        self.opened = False
        # Guards the shared sqlite3.Connection against concurrent tool
        # invocations. The vault's `flock` (see `_acquire_lock`) excludes a
        # second *process*; it says nothing about threads inside this one,
        # and the MCP runtime dispatches every synchronous tool call onto an
        # anyio worker thread (see mindpalace/index/db.py's
        # check_same_thread=False), so two `tools/call` requests pipelined by
        # one client can legitimately run concurrently on two different
        # threads. `server.py` is the sole place transport-originated calls
        # enter the vault, and takes this lock around every tool dispatch.
        # Created here rather than in `open()` so it exists even if `open()`
        # never succeeds.
        self.lock = threading.RLock()

    # ---- lifecycle ----------------------------------------------------

    def open(self) -> "Session":
        self.paths, self.config = open_vault(self._root, init=self._init)
        self._acquire_lock()
        try:
            self.store = VaultStore(self.paths)
            self._open_cache()
            self.oplog = OpLog(self.paths.op_log)
            self.decisions = DecisionLog(self.paths.decisions_log)
            self.embedder = self._explicit_embedder or get_embedder(self.config.embedder)
            self.opened = True
            self._verify_cache_model()
            self.heal()
        except BaseException:
            # The lock was taken above; if anything after it fails (bad
            # embedder config, a corrupt cache file, ...) the lock must not
            # outlive this failed open, or a vault that failed to open once
            # could never be opened again for the rest of the process.
            self.opened = False
            conn = getattr(self, "conn", None)
            if conn is not None:
                conn.close()
            self._release_lock()
            raise
        return self

    def close(self) -> None:
        if self.opened:
            self.conn.close()
            self._release_lock()
            self.opened = False

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _open_cache(self) -> None:
        """Connect to `.graph/mindpalace.db`, recovering from corruption.

        Tier 3 is delete-and-rebuild by construction (spec §10: "Corrupt
        SQLite -- detected on open, rebuilt from Tier 1. Nothing is lost.").
        `sqlite3` does not reject a corrupt file at `connect()` time -- the
        error only surfaces on the first statement, which `create_schema`
        issues -- so both are covered by the same `try`. Recovery just
        deletes the file and starts over: `_verify_cache_model` (called
        right after this, in `open()`) finds no `cache_meta` row in the
        fresh database and calls `resync()`, which rebuilds everything from
        Tier 1 with no further help needed here.
        """
        try:
            self.conn = db.connect(self.paths.graph_db)
            db.create_schema(self.conn)
        except sqlite3.DatabaseError:
            conn = getattr(self, "conn", None)
            if conn is not None:
                conn.close()
            self.paths.graph_db.unlink(missing_ok=True)
            self.conn = db.connect(self.paths.graph_db)
            db.create_schema(self.conn)

    def _acquire_lock(self) -> None:
        """Exclusive advisory lock held for the session's lifetime.

        An `exists()` check followed by a write would be check-then-write: two
        servers starting together could both see no lock and both proceed.
        `flock` is atomic, and the kernel releases it if we die, which also makes
        a stale lock from a crashed process self-healing.
        """
        lock = self.paths.lock
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = os.read(self._lock_fd, 32).decode(errors="replace").strip()
            os.close(self._lock_fd)
            self._lock_fd = None
            raise VaultLockedError(
                f"vault {self.paths.root} is already open (pid {holder or 'unknown'})"
            ) from exc
        os.ftruncate(self._lock_fd, 0)
        os.write(self._lock_fd, str(os.getpid()).encode())
        os.fsync(self._lock_fd)

    def _release_lock(self) -> None:
        if getattr(self, "_lock_fd", None) is None:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None
        self.paths.lock.unlink(missing_ok=True)

    def _verify_cache_model(self) -> None:
        """Rebuild when the embedder changed (spec §5).

        Vectors are filtered by `model_id`, so a swapped model does not error —
        it returns nothing, which reads as an empty vault. Detect and rebuild.
        """
        if db.read_meta(self.conn) == (self.embedder.model_id, self.embedder.dim):
            return
        self.resync()

    # ---- operations ---------------------------------------------------

    @contextmanager
    def operation(self, intent: dict):
        op_id = self.oplog.begin(intent)
        yield op_id
        self.oplog.commit(op_id)

    def heal(self) -> bool:
        """Reconcile after a crash or an external edit. Idempotent."""
        pending = self.oplog.pending()
        drifted = has_drift(self.conn, self.store)
        if not pending and not drifted:
            return False
        if drifted:
            # MINDPALACE.md is in the drift set, and its edge_types drive how the
            # fold groups aggregates — so reload before re-deriving anything.
            self.config = load_config(self.paths.mindpalace_md)
        self.resync()
        for record in pending:
            self.oplog.commit(record["op"])
        return True

    def resync(self) -> None:
        sync(self.conn, self.store, self.config, self.embedder, self.statuses())

    def statuses(self) -> dict[str, str]:
        return self.decisions.status_map()
