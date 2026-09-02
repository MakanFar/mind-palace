"""Process-level vault session: locking, crash healing, operation framing."""

from __future__ import annotations

import errno
import fcntl
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from mindpalace.config import Config, load_config, open_vault
from mindpalace.embed import Embedder, get_embedder
from mindpalace.index import db
from mindpalace.index.sync import has_drift, sync
from mindpalace.oplog import DecisionLog, MergeLog, OpLog, VocabularyLog
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


class VaultLockedError(RuntimeError):
    """Raised when another server holds the vault's write lock."""


#: How long a write waits for another session's write before giving up. Long
#: enough to sit out a `rebuild` or a `cluster` in the other session, short
#: enough that a wedged holder surfaces as an error rather than a hang.
DEFAULT_LOCK_TIMEOUT = 60.0

#: `flock` has no timed variant, so the wait is a poll. Short enough that an
#: uncontended handoff is imperceptible, long enough not to spin a core.
LOCK_POLL_SECONDS = 0.02


class Session:
    def __init__(
        self,
        root: Path,
        *,
        init: bool = False,
        embedder: Embedder | None = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> None:
        self._root = Path(root)
        self._init = init
        self._explicit_embedder = embedder
        self._lock_fd: int | None = None
        self._write_depth = 0
        self.lock_timeout = lock_timeout
        self.paths: VaultPaths
        self.config: Config
        self.store: VaultStore
        self.embedder: Embedder
        self.opened = False
        # Guards the shared sqlite3.Connection against concurrent tool
        # invocations. The vault's `flock` (see `write_lock`) excludes a
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
        self._open_lock_file()
        try:
            self.store = VaultStore(self.paths)
            self._open_cache()
            self.oplog = OpLog(self.paths.op_log)
            self.decisions = DecisionLog(self.paths.decisions_log)
            self.vocabulary = VocabularyLog(self.paths.vocabulary_log)
            self.merges = MergeLog(self.paths.merges_log)
            self.embedder = self._explicit_embedder or get_embedder(self.config.embedder)
            self.opened = True
            self._verify_cache_model()
            self.heal()
        except BaseException:
            # A failed open must leave nothing behind: the descriptor opened
            # above, and any connection made after it, both belong to a
            # session that does not exist.
            self.opened = False
            conn = getattr(self, "conn", None)
            if conn is not None:
                conn.close()
            self._close_lock_file()
            raise
        return self

    def close(self) -> None:
        if self.opened:
            self.conn.close()
            self._close_lock_file()
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

    def _open_lock_file(self) -> None:
        """Open `.mindpalace/lock`, without locking it.

        Opening a vault is a read, and hosts start one server process per
        editor session -- an exclusive lock taken here would let exactly one
        of those sessions work and fail all the others at startup. The
        descriptor lives as long as the session because a `flock` belongs to
        the open file description: reopening the file per write would give
        each write a lock nobody else can see.
        """
        lock = self.paths.lock
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)

    def _close_lock_file(self) -> None:
        """Close our descriptor. Deliberately does not unlink the file.

        Unlinking it was a way to hand a second writer a *different* inode:
        a session still holding the flock keeps it on the now-unnamed file,
        while a newcomer creates a fresh one at the same path and locks that.
        Both then believe they are the only writer. The file is small, and a
        leftover one costs nothing -- `flock` state lives in the kernel, not
        in its contents.
        """
        if getattr(self, "_lock_fd", None) is None:
            return
        os.close(self._lock_fd)
        self._lock_fd = None
        self._write_depth = 0

    @contextmanager
    def write_lock(self):
        """Hold the vault's single-writer lock for the duration of a write.

        Two layers, because there are two kinds of concurrent writer:
        `self.lock` (an RLock) excludes other threads in this process -- the
        MCP runtime runs tool bodies on a worker pool -- and the `flock`
        excludes other processes. Both are reentrant, which they have to be:
        `resync()` locks, and it is called from inside `operation()` (see
        tools.py), so a non-reentrant release in the inner scope would unlock
        the vault mid-write.

        `flock` has no timed variant, hence the poll. The kernel drops the
        lock if the holder dies, so a crashed writer never wedges the vault.
        """
        with self.lock:
            self._acquire_write_lock()
            try:
                yield
            finally:
                self._release_write_lock()

    def _acquire_write_lock(self) -> None:
        if self._write_depth == 0:
            if self._lock_fd is None:
                raise VaultLockedError("session is not open")
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                        # Only "someone else holds it" is worth retrying. A
                        # closed or invalid descriptor would otherwise burn
                        # the whole timeout and then blame an innocent pid.
                        raise
                    if time.monotonic() >= deadline:
                        raise VaultLockedError(
                            f"vault {self.paths.root} is already open "
                            f"(pid {self._recorded_pid()})"
                        ) from exc
                    time.sleep(LOCK_POLL_SECONDS)
            # Diagnostics only, so no fsync: whoever reads this file is a human
            # looking at an error message, and the lock itself is the flock.
            # Positional I/O because the descriptor outlives one write and a
            # drifting file offset would append pids instead of replacing them.
            os.ftruncate(self._lock_fd, 0)
            os.pwrite(self._lock_fd, str(os.getpid()).encode(), 0)
        self._write_depth += 1

    def _release_write_lock(self) -> None:
        self._write_depth -= 1
        if self._write_depth == 0 and self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _recorded_pid(self) -> str:
        recorded = os.pread(self._lock_fd, 32, 0).decode(errors="replace")
        return recorded.strip() or "unknown"

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
        """Frame one mutation: exclusive, logged, committed on success.

        The lock is taken before the op log is written, not after. A write
        that never got the lock never started, so it must not leave a `begin`
        with no `commit` behind -- the next open would replay it as a crash.
        """
        with self.write_lock():
            op_id = self.oplog.begin(intent)
            yield op_id
            self.oplog.commit(op_id)

    def heal(self) -> bool:
        """Reconcile after a crash or an external edit. Idempotent.

        Deciding whether there is work is a read, and it is what every open
        does; only the repair takes the lock. A session that healed nothing
        must not have waited on another session's `rebuild` to discover that.
        The check is then repeated under the lock, because the wait is exactly
        the window in which the other session may have healed the same drift.
        """
        if not self.oplog.pending() and not has_drift(self.conn, self.store):
            return False
        with self.write_lock():
            pending = self.oplog.pending()
            drifted = has_drift(self.conn, self.store)
            if not pending and not drifted:
                return False
            if drifted:
                # MINDPALACE.md is in the drift set, and its edge_types drive how
                # the fold groups aggregates — so reload before re-deriving.
                self.config = load_config(self.paths.mindpalace_md)
            self.resync()
            for record in pending:
                self.oplog.commit(record["op"])
            return True

    def resync(self) -> None:
        with self.write_lock():
            sync(
                self.conn,
                self.store,
                self.config,
                self.embedder,
                self.statuses(),
                **self.overlays(),
            )

    def statuses(self) -> dict[str, str]:
        return self.decisions.status_map()

    def overlays(self) -> dict:
        """The folded vocabulary and merge logs, as keyword arguments for
        `fold` / `sync` / `rebuild` (docs/decisions/0001 §1, §5)."""
        return {
            "adoptions": self.vocabulary.adoptions(),
            "merges": self.merges.merges(),
            "kept": self.merges.kept(),
        }
