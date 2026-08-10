"""Vector storage and brute-force cosine search.

Deliberate deviation from the spec's `sqlite-vec`: at MVP scale a numpy scan
over a few hundred vectors is sub-millisecond, and this avoids a loadable
extension that is fragile across macOS Python builds. Swapping in `sqlite-vec`
later touches only this module.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

DTYPE = np.float32


def _normalise(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=DTYPE).ravel()
    norm = float(np.linalg.norm(array))
    return array if norm == 0.0 else array / norm


def store(
    conn: sqlite3.Connection,
    doc_id: str,
    kind: str,
    model_id: str,
    vector: np.ndarray,
) -> None:
    unit = _normalise(vector)
    conn.execute(
        "INSERT INTO vectors (doc_id, kind, model_id, dim, vector) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET kind=excluded.kind, "
        "model_id=excluded.model_id, dim=excluded.dim, vector=excluded.vector",
        (doc_id, kind, model_id, int(unit.size), unit.tobytes()),
    )


def search(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    model_id: str,
    kinds: Sequence[str],
    limit: int,
) -> list[tuple[str, float]]:
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT doc_id, vector FROM vectors "
        f"WHERE model_id = ? AND kind IN ({placeholders})",
        (model_id, *kinds),
    ).fetchall()
    if not rows:
        return []

    query = _normalise(query_vector)
    matrix = np.vstack([np.frombuffer(row["vector"], dtype=DTYPE) for row in rows])
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[:limit]
    return [(rows[index]["doc_id"], float(scores[index])) for index in order]


def clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM vectors")
