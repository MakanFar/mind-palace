"""Regenerate Tier 3 and mark Tier 2 stale. Never deletes or rewrites prose."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from collections.abc import Iterable

from mindpalace.atomic import content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.graph.fold import FoldedEntity, GraphTables, fold
from mindpalace.ids import slugify
from mindpalace.index.sync import sync
from mindpalace.models import EntityPage, Note
from mindpalace.vault.store import VaultStore


@dataclass(frozen=True)
class RebuildReport:
    synced: int
    pages_written: int
    pages_marked_stale: int
    reports_marked_stale: int


def entity_input_hash(
    entity: FoldedEntity, tables: GraphTables, notes_by_id: dict[str, Note]
) -> str:
    """Hash the *evidence* a description was written from, not its identity.

    Hashing ids alone (type, note ids, edge shape) would leave a page reading
    `stale: false` after its source note was rewritten in place -- the id never
    changes, so the hash never moves, and the prose silently stops describing
    anything real. Everything a writer could have read has to be in here.

    That includes every relationship assertion touching this entity, not only
    the ones sitting inside a *traversable* (i.e. confirmed) aggregate: a
    description writer can read a proposed or dismissed assertion's rationale
    just as easily as a confirmed one, so restricting evidence to confirmed
    aggregates would leave a page reading fresh forever if its only assertion
    is still awaiting review and someone edits its rationale in place.
    """
    parts = [f"type={entity.type}"]

    for note_id in sorted(entity.note_ids):
        note = notes_by_id.get(note_id)
        if note is None:
            continue
        parts.append(f"note={note_id}:{content_hash(note.body)}")
        for instance in note.entities:
            if slugify(instance.name) == entity.slug:
                parts.append(f"instance={content_hash(instance.description)}")

    for assertion in sorted(tables.assertions.values(), key=lambda item: item.id):
        if entity.slug not in (assertion.source, assertion.target):
            continue
        parts.append(
            f"assertion={assertion.id}:{assertion.status}:{assertion.strength}:"
            f"{content_hash(assertion.description)}"
        )

    for aggregate in sorted(tables.aggregates.values(), key=lambda item: item.key):
        if not aggregate.traversable:
            continue
        if entity.slug not in (aggregate.source, aggregate.target):
            continue
        parts.append(
            f"edge={aggregate.key}:{aggregate.weight}:{aggregate.mean_strength}"
        )

    for claim in sorted(tables.claims.values(), key=lambda item: item.id):
        if claim.subject == entity.slug:
            parts.append(f"claim={claim.id}:{claim.status}:{content_hash(claim.text)}")

    return content_hash("\n".join(parts))


def community_input_hash(members: Iterable[str], tables: GraphTables) -> str:
    """Hash the community's evidence, not just its member slugs.

    Membership can hold steady while every description, weight, and claim under
    it changes -- a report hashed on slugs alone would never notice.
    """
    membership = set(members)
    parts = []

    for slug in sorted(membership):
        entity = tables.entities.get(slug)
        parts.append(
            f"entity={slug}" if entity is None else f"entity={slug}:{entity.type}:{entity.rank}"
        )

    for aggregate in sorted(tables.aggregates.values(), key=lambda item: item.key):
        if (
            aggregate.traversable
            and aggregate.source in membership
            and aggregate.target in membership
        ):
            parts.append(
                f"edge={aggregate.key}:{aggregate.weight}:{aggregate.mean_strength}"
            )

    for claim in sorted(tables.claims.values(), key=lambda item: item.id):
        if claim.subject in membership and claim.status == "confirmed":
            parts.append(f"claim={claim.id}:{content_hash(claim.text)}")

    return content_hash("\n".join(parts))


def mark_stale_reports(
    conn: sqlite3.Connection, store: VaultStore, tables: GraphTables
) -> int:
    """Flag any report whose community evidence has moved since it was written.

    Shared by `rebuild` and `cluster_tool`: reclustering must not leave an
    obsolete report reading as `present` in global_search until some unrelated
    rebuild happens to run.
    """
    membership = {
        row["lineage_id"]: (row["members"].split(",") if row["members"] else [])
        for row in conn.execute("SELECT lineage_id, members FROM communities")
    }
    marked = 0
    for report in list(store.iter_reports()):
        members = membership.get(report.lineage_id)
        if members is None:
            continue
        expected = community_input_hash(members, tables)
        if report.input_hash != expected and not report.stale:
            report.stale = True
            report.input_hash = expected
            store.write_report(report)
            marked += 1
    return marked


def _related_lines(slug: str, tables: GraphTables) -> list[str]:
    lines = []
    for aggregate in tables.aggregates.values():
        if not aggregate.traversable:
            continue
        if aggregate.source == slug:
            lines.append(f"{aggregate.type} [[{aggregate.target}]]")
        elif aggregate.target == slug:
            lines.append(f"{aggregate.type} [[{aggregate.source}]]")
    return sorted(set(lines))


def rebuild(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    embedder: Embedder,
    statuses: dict[str, str],
    scope: str = "all",
) -> RebuildReport:
    synced = 0
    if scope in {"cache", "all"}:
        synced = sync(conn, store, config, embedder, statuses).notes

    notes = list(store.iter_notes())
    notes_by_id = {note.id: note for note in notes}
    tables = fold(notes, statuses, config)

    written = 0
    marked_stale = 0
    if scope in {"related_blocks", "all"}:
        for entity in tables.entities.values():
            expected = entity_input_hash(entity, tables, notes_by_id)
            existing = store.read_entity_page(entity.slug)
            related = _related_lines(entity.slug, tables)

            if existing is None:
                store.write_entity_page(
                    EntityPage(
                        slug=entity.slug,
                        type=entity.type,
                        description="",
                        generated_from=list(entity.note_ids),
                        input_hash=expected,
                        stale=True,
                        related=related,
                    )
                )
                written += 1
                continue

            became_stale = existing.input_hash != expected
            if became_stale and not existing.stale:
                marked_stale += 1

            store.write_entity_page(
                EntityPage(
                    slug=entity.slug,
                    type=existing.user.get("type", entity.type),
                    description=existing.description,
                    generated_from=list(entity.note_ids),
                    input_hash=expected,
                    stale=existing.stale or became_stale,
                    user=existing.user,
                    related=related,
                )
            )
            written += 1

    reports_marked = mark_stale_reports(conn, store, tables)

    # Entity pages are written *after* the first sync, so their docs and vectors
    # would otherwise lag a full cycle behind. Re-sync once they exist.
    if written or reports_marked:
        sync(conn, store, config, embedder, statuses)

    return RebuildReport(
        synced=synced,
        pages_written=written,
        pages_marked_stale=marked_stale,
        reports_marked_stale=reports_marked,
    )
