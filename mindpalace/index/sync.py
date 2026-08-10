"""Full rebuild of the Tier 3 cache from Tier 1 sources."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from mindpalace.atomic import atomic_write, content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.graph.fold import (
    DuplicateAssertionIdError,
    GraphTables,
    UnknownDecisionActionError,
    UnknownEdgeTypeError,
    fold,
)
from mindpalace.ids import slugify
from mindpalace.index import db, vectors
from mindpalace.models import Capture, Note, capture_from_markdown, note_from_markdown
from mindpalace.vault.store import VaultStore

DOC_KINDS = ("capture", "note", "entity", "report", "unparsed")


@dataclass(frozen=True)
class SyncReport:
    notes: int
    captures: int
    entities: int
    aggregates: int
    issues: tuple[tuple[str, str, str], ...] = field(default=())


def _relative(store: VaultStore, path: Path) -> str:
    return str(path.relative_to(store.paths.root))


def _iter_source_files(store: VaultStore) -> Iterator[Path]:
    """Every file the derived cache depends on.

    Includes the decision log and MINDPALACE.md. Both are inputs — decisions
    determine assertion status, config determines how the fold groups edges — so
    omitting them means an external edit to either leaves the cache silently
    stale with nothing able to notice.
    """
    yield from sorted(store.paths.captures.glob("*.md"))
    yield from sorted(store.paths.notes.glob("*.md"))
    for dependency in (store.paths.decisions_log, store.paths.mindpalace_md):
        if dependency.exists():
            yield dependency


def _load_sources(
    store: VaultStore,
) -> tuple[
    list[Capture],
    list[Note],
    list[tuple[str, str, str]],
    list[tuple[str, str]],
    dict[str, tuple[str, str]],
]:
    """Parse every source file, degrading rather than dropping on failure.

    Returns captures, successfully-parsed notes, issues recorded so far,
    (path, raw_text) pairs for files that failed to parse at all, and a
    note_id -> (path, raw_text) map for every note that DID parse -- kept
    around so that a note `fold` later rejects (see `_fold_with_quarantine`)
    can still be indexed as a degraded document by its path, without
    re-reading the file from disk.
    """
    captures: list[Capture] = []
    notes: list[Note] = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []
    note_sources: dict[str, tuple[str, str]] = {}

    for path in sorted(store.paths.captures.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            captures.append(capture_from_markdown(raw))
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_capture", str(exc)))
            degraded.append((_relative(store, path), raw))

    for path in sorted(store.paths.notes.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            note = note_from_markdown(raw)
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_note", str(exc)))
            degraded.append((_relative(store, path), raw))
            continue
        notes.append(note)
        note_sources[note.id] = (_relative(store, path), raw)

    return captures, notes, issues, degraded, note_sources


def _fold_with_quarantine(
    notes: list[Note],
    statuses: dict[str, str],
    config: Config,
    issues: list[tuple[str, str, str]],
    note_paths: dict[str, str],
) -> tuple[GraphTables, set[str]]:
    """Fold, quarantining and retrying on a typed `FoldError` instead of
    letting it propagate out of `sync`.

    `fold` (Task 9) now raises rather than silently misbehaving on a
    malformed source set. That is exactly right for `fold`'s own contract,
    but wrong for `sync`: spec §10 requires one bad note can never take the
    whole cache rebuild -- and therefore the whole vault -- down. So `sync`
    must not call `fold` bare. Instead: try, and on a recognised error,
    remove the offending note (or status-log entry) from the candidate set
    and retry, recording a `vault_issue` for what was excluded.

    Mutates `issues` in place. Returns the tables folded from whatever
    remained after quarantine, plus the set of note ids that got excluded
    so the caller can still index them as degraded FTS documents (a note
    the graph rejected must stay findable so the user can go fix it).
    """
    candidates: dict[str, Note] = {note.id: note for note in notes}
    # Never mutate the caller's statuses dict -- it may be reused elsewhere
    # (e.g. re-synced after a fix) and quarantine is a `sync`-local decision.
    status_map = dict(statuses)
    excluded: set[str] = set()

    # Each accepted iteration strictly shrinks len(candidates) + len(status_map)
    # by at least one, so this many attempts is always enough for a set of
    # exceptions that shrink correctly. A future exception type (or a bug)
    # that doesn't shrink on would otherwise spin forever; bound it instead.
    max_iterations = len(notes) + len(statuses) + 1

    for _ in range(max_iterations):
        try:
            return fold(candidates.values(), status_map, config), excluded
        except UnknownEdgeTypeError as exc:
            path = note_paths.get(exc.note_id, f"notes/{exc.note_id}")
            issues.append(
                (
                    path,
                    "unknown_edge_type",
                    f"edge type {exc.edge_type!r} is not declared in "
                    f"MINDPALACE.md (assertion {exc.assertion_id})",
                )
            )
            excluded.add(exc.note_id)
            candidates.pop(exc.note_id, None)
        except DuplicateAssertionIdError as exc:
            keep, *rest = exc.note_ids
            for note_id in rest:
                path = note_paths.get(note_id, f"notes/{note_id}")
                issues.append(
                    (
                        path,
                        "duplicate_assertion_id",
                        f"assertion {exc.assertion_id!r} collides with the one "
                        f"already recorded from note {keep!r}",
                    )
                )
                excluded.add(note_id)
                candidates.pop(note_id, None)
        except UnknownDecisionActionError as exc:
            issues.append(
                (
                    ".mindpalace/decisions.jsonl",
                    "unknown_decision_action",
                    f"action {exc.action!r} is not recognised "
                    f"(assertion {exc.assertion_id})",
                )
            )
            status_map = {
                item_id: action
                for item_id, action in status_map.items()
                if item_id != exc.assertion_id
            }

    raise RuntimeError(
        f"fold quarantine loop exceeded its iteration cap ({max_iterations}) "
        "without converging -- a FoldError subtype is being raised for "
        "something the loop cannot shrink on. This is an internal bug, not "
        "a vault problem."
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def _ambiguous_alias_issues(
    store: VaultStore, entity_pages: list
) -> list[tuple[str, str, str]]:
    """Two entities claiming the same alias is a vault_issue (spec §8.5),
    never silently resolved in favour of one -- aliases explicitly do not
    merge identities. `_resolve_slug` (mindpalace/tools.py) raises on lookup
    when it hits one of these live; this is the same collision surfaced
    proactively through `review_queue` so it can be noticed and fixed even
    before anyone happens to look the ambiguous name up.
    """
    owners: dict[str, list[str]] = {}
    for page in entity_pages:
        for alias in page.user.get("aliases", []):
            owners.setdefault(slugify(alias), []).append(page.slug)

    issues: list[tuple[str, str, str]] = []
    for alias, slugs in sorted(owners.items()):
        if len(slugs) <= 1:
            continue
        claimants = sorted(set(slugs))
        paths = ", ".join(
            _relative(store, store.paths.entity_path(slug)) for slug in claimants
        )
        issues.append(
            (
                paths,
                "ambiguous_alias",
                f"alias {alias!r} is claimed by more than one entity: "
                f"{claimants}; aliases do not merge identities",
            )
        )
    return issues


def _render_index(
    captures: list[Capture], notes: list[Note], tables: GraphTables
) -> str:
    """The human- and Obsidian-readable catalog promised by spec §4.5."""
    lines = [
        "# Index",
        "",
        f"{len(captures)} captures · {len(notes)} notes · "
        f"{len(tables.entities)} entities",
        "",
        "## Notes",
        "",
    ]
    for note in sorted(notes, key=lambda item: (item.created, item.id)):
        lines.append(f"- `{note.id}` {note.created[:10]} — {_first_line(note.body)}")
    lines += ["", "## Entities", ""]
    for entity in sorted(
        tables.entities.values(), key=lambda item: (-item.rank, item.slug)
    ):
        lines.append(f"- [[{entity.slug}]] — {entity.type}, rank {entity.rank}")
    return "\n".join(lines) + "\n"


def sync(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    embedder: Embedder,
    statuses: dict[str, str],
) -> SyncReport:
    captures, notes, issues, degraded, note_sources = _load_sources(store)

    note_paths = {note_id: path for note_id, (path, _raw) in note_sources.items()}
    tables, excluded_notes = _fold_with_quarantine(
        notes, statuses, config, issues, note_paths
    )

    # A note the graph rejected (unknown edge type, duplicate assertion id)
    # must still be findable so the user can go fix it -- spec §10 applies
    # to graph-level rejections exactly as it does to parse failures.
    folded_notes = [note for note in notes if note.id not in excluded_notes]
    for note_id in excluded_notes:
        degraded.append(note_sources[note_id])

    documents: list[tuple[str, str, str, str]] = []
    for capture in captures:
        documents.append((capture.id, "capture", _first_line(capture.text), capture.text))
    for note in folded_notes:
        # Assertion descriptions ride along in the note's searchable text rather
        # than becoming their own documents: they must be findable (spec §8.1)
        # but an `x_` id is not something `read` can return.
        assertion_text = " ".join(
            a.description for a in note.relationship_assertions
        )
        claim_text = " ".join(c.text for c in note.claim_assertions)
        searchable = "\n".join(filter(None, [note.body, assertion_text, claim_text]))
        documents.append((note.id, "note", _first_line(note.body), searchable))
    entity_pages = list(store.iter_entity_pages())
    for page in entity_pages:
        documents.append((f"e_{page.slug}", "entity", page.slug, page.description))
    issues.extend(_ambiguous_alias_issues(store, entity_pages))
    for report in store.iter_reports():
        documents.append(
            (report.lineage_id, "report", report.title, f"{report.summary}\n{report.findings}")
        )
    # A file we cannot parse -- or a note the graph rejected -- is still
    # findable, not vanished. Keyed by path, since it has no usable id.
    for relative_path, raw in degraded:
        documents.append((relative_path, "unparsed", relative_path, raw))

    # Embed before opening the transaction. The model call is the slow part and
    # must not hold a write transaction open across it.
    matrix = embedder.embed([text for _, _, _, text in documents]) if documents else []

    with conn:
        for table in (
            "files",
            "entities",
            "entity_sources",
            "assertions",
            "aggregates",
            "aggregate_members",
            "claims",
            "docs",
            "vault_issues",
        ):
            conn.execute(f"DELETE FROM {table}")
        vectors.clear(conn)
        db.write_meta(conn, embedder.model_id, embedder.dim)

        for path in _iter_source_files(store):
            conn.execute(
                "INSERT INTO files (path, hash) VALUES (?, ?)",
                (_relative(store, path), content_hash(path.read_text(encoding="utf-8"))),
            )

        for entity in tables.entities.values():
            conn.execute(
                "INSERT INTO entities (slug, type, rank) VALUES (?, ?, ?)",
                (entity.slug, entity.type, entity.rank),
            )
            conn.executemany(
                "INSERT INTO entity_sources (slug, note_id) VALUES (?, ?)",
                [(entity.slug, note_id) for note_id in entity.note_ids],
            )

        for assertion in tables.assertions.values():
            conn.execute(
                "INSERT INTO assertions (id, kind, note_id, source, target, type, "
                "strength, description, status) VALUES (?, 'relationship', ?, ?, ?, ?, ?, ?, ?)",
                (
                    assertion.id,
                    assertion.note_id,
                    assertion.source,
                    assertion.target,
                    assertion.type,
                    assertion.strength,
                    assertion.description,
                    assertion.status,
                ),
            )

        for claim in tables.claims.values():
            conn.execute(
                "INSERT INTO claims (id, note_id, subject, text, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (claim.id, claim.note_id, claim.subject, claim.text, claim.status),
            )

        for aggregate in tables.aggregates.values():
            conn.execute(
                "INSERT INTO aggregates (key, source, type, target, weight, "
                "mean_strength, traversable) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    aggregate.key,
                    aggregate.source,
                    aggregate.type,
                    aggregate.target,
                    aggregate.weight,
                    aggregate.mean_strength,
                    int(aggregate.traversable),
                ),
            )
            conn.executemany(
                "INSERT INTO aggregate_members (key, assertion_id) VALUES (?, ?)",
                [(aggregate.key, aid) for aid in aggregate.assertion_ids],
            )

        conn.executemany(
            "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
            documents,
        )
        conn.executemany(
            "INSERT INTO vault_issues (path, kind, detail) VALUES (?, ?, ?)", issues
        )

        for (doc_id, kind, _, _), vector in zip(documents, matrix, strict=True):
            vectors.store(conn, doc_id, kind, embedder.model_id, vector)

    atomic_write(store.paths.index_md, _render_index(captures, folded_notes, tables))

    return SyncReport(
        notes=len(folded_notes),
        captures=len(captures),
        entities=len(tables.entities),
        aggregates=len(tables.aggregates),
        issues=tuple(issues),
    )


def has_drift(conn: sqlite3.Connection, store: VaultStore) -> bool:
    recorded = {
        row["path"]: row["hash"] for row in conn.execute("SELECT path, hash FROM files")
    }
    seen: set[str] = set()
    for path in _iter_source_files(store):
        relative = _relative(store, path)
        seen.add(relative)
        current = content_hash(path.read_text(encoding="utf-8"))
        if recorded.get(relative) != current:
            return True
    return seen != recorded.keys()
