"""Crash-safe single-file writes with optimistic concurrency control."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class ConflictError(RuntimeError):
    """Raised when a file changed between read and write."""


def content_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def atomic_write(path: Path, content: str) -> None:
    """Write via temp file + rename so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def cas_write(path: Path, content: str, expected_hash: str | None) -> None:
    """Write only if the on-disk content still hashes to `expected_hash`.

    `expected_hash=None` asserts the file does not yet exist. A mismatch aborts
    rather than clobbering an edit made in Obsidian since we last read.
    """
    if expected_hash is None:
        # Exclusive create via atomic OS link: write to temp, link atomically.
        # This avoids the race window between exists() check and replace().
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                raise ConflictError(f"{path} already exists")
        except ConflictError:
            temp_path.unlink(missing_ok=True)
            raise
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        else:
            temp_path.unlink()
    else:
        if not path.exists():
            raise ConflictError(f"{path} changed on disk: expected content, found none")
        actual = content_hash(path.read_text(encoding="utf-8"))
        if actual != expected_hash:
            raise ConflictError(
                f"{path} changed on disk: expected {expected_hash}, found {actual}"
            )
        atomic_write(path, content)
