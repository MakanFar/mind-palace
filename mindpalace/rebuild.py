"""Regenerate Tier 3 and mark Tier 2 stale. Never deletes or rewrites prose."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from collections.abc import Iterable, Mapping

from mindpalace.atomic import ConflictError, content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.frontmatter import FrontMatterError
from mindpalace.graph.fold import FoldedEntity, GraphTables
from mindpalace.ids import slugify
from mindpalace.index.sync import _load_notes, fold_notes_with_quarantine, sync
from mindpalace.models import EntityPage, Note
from mindpalace.vault.store import VaultStore

# Kinds `fold_notes_with_quarantine` can produce. Scoped so `rebuild` can
# persist its own quarantine pass without disturbing vault_issue kinds that
# belong to a different stage (ambiguous_alias, stale_prose, ...), and
# without duplicating whatever `sync` already wrote this same call when
# scope includes "cache" -- both passes fold the same source set, so a
# kind-scoped replace is idempotent either way.
_FOLD_ISSUE_KINDS = (
    "malformed_note",
    "unknown_edge_type",
    "duplicate_assertion_id",
    "unknown_decision_action",
    "merge_cycle",
)


@dataclass(frozen=True)
class RebuildReport:
    synced: int
    pages_written: int
    pages_marked_stale: int
    reports_marked_stale: int


def _units_suffix(unit_ids: tuple[str, ...]) -> str:
    """Only items that carry provenance change the hash text. Appending an
    empty suffix would have flipped every page and report stale on the
    first rebuild after the upgrade that introduced text units."""
    return f":{','.join(unit_ids)}" if unit_ids else ""


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
    # Descriptions recorded under a name since merged into this entity are
    # evidence a writer read too (docs/decisions/0001 §5).
    names = {entity.slug, *entity.merged_from}

    for note_id in sorted(entity.note_ids):
        note = notes_by_id.get(note_id)
        if note is None:
            continue
        parts.append(f"note={note_id}:{content_hash(note.body)}")
        for instance in note.entities:
            if slugify(instance.name) in names:
                parts.append(f"instance={content_hash(instance.description)}")

    for assertion in sorted(tables.assertions.values(), key=lambda item: item.id):
        if entity.slug not in (assertion.source, assertion.target):
            continue
        parts.append(
            f"assertion={assertion.id}:{assertion.status}:{assertion.strength}:"
            f"{assertion.type}:{assertion.proposed_type}:"
            f"{content_hash(assertion.description)}{_units_suffix(assertion.text_unit_ids)}"
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
            parts.append(
                f"claim={claim.id}:{claim.status}:{claim.valid_from}:{claim.valid_to}:"
                f"{content_hash(claim.text)}{_units_suffix(claim.text_unit_ids)}"
            )

    return content_hash("\n".join(parts))


def community_input_hash(
    members: Iterable[str], tables: GraphTables, notes_by_id: dict[str, Note]
) -> str:
    """Hash the community's evidence, not just its member slugs.

    Membership can hold steady while every description, weight, and claim under
    it changes -- a report hashed on slugs alone would never notice.

    Mirrors `entity_input_hash`'s coverage, scoped to the community's members:
    a report writer can read a member's raw notes (entity-instance
    descriptions, assertion rationale, claim text) exactly as freely as it can
    read the member's own entity page -- via `read` or `get_entity` on any
    member before composing the report. So this covers, for every member:
    note bodies and entity-instance descriptions; every assertion touching
    that member regardless of confirmation status (status, strength,
    description); and every claim whose subject is that member regardless of
    status. The confirmed-only, both-endpoints-inside aggregate summary is
    kept alongside as the community's internal-structure signal, not as a
    substitute for the assertion-level evidence above.
    """
    membership = set(members)
    parts = []

    for slug in sorted(membership):
        entity = tables.entities.get(slug)
        parts.append(
            f"entity={slug}" if entity is None else f"entity={slug}:{entity.type}:{entity.rank}"
        )
        if entity is None:
            continue
        names = {slug, *entity.merged_from}
        for note_id in sorted(entity.note_ids):
            note = notes_by_id.get(note_id)
            if note is None:
                continue
            parts.append(f"note={slug}:{note_id}:{content_hash(note.body)}")
            for instance in note.entities:
                if slugify(instance.name) in names:
                    parts.append(f"instance={slug}:{content_hash(instance.description)}")

    for assertion in sorted(tables.assertions.values(), key=lambda item: item.id):
        if membership.isdisjoint((assertion.source, assertion.target)):
            continue
        parts.append(
            f"assertion={assertion.id}:{assertion.status}:{assertion.strength}:"
            f"{content_hash(assertion.description)}{_units_suffix(assertion.text_unit_ids)}"
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
        if claim.subject in membership:
            parts.append(
                f"claim={claim.id}:{claim.status}:{claim.valid_from}:{claim.valid_to}:"
                f"{content_hash(claim.text)}{_units_suffix(claim.text_unit_ids)}"
            )

    return content_hash("\n".join(parts))


def mark_stale_reports(
    conn: sqlite3.Connection, store: VaultStore, tables: GraphTables
) -> int:
    """Flag any report whose community evidence has moved since it was written.

    Shared by `rebuild` and `cluster_tool`: reclustering must not leave an
    obsolete report reading as `present` in global_search until some unrelated
    rebuild happens to run.

    Its own signature stays `(conn, store, tables)` -- callers already built
    against it must not need updating -- so the `notes_by_id` that
    `community_input_hash` needs is built locally from `store`, which was
    already a parameter here.
    """
    membership = {
        row["lineage_id"]: (row["members"].split(",") if row["members"] else [])
        for row in conn.execute("SELECT lineage_id, members FROM communities")
    }
    # `_load_notes`, not `store.iter_notes()`: a note that fails to parse
    # must not take clustering/rebuild down just to compute a staleness
    # hash (spec §10). Issues are already recorded by whichever quarantine
    # pass (`sync`, `fold_notes_with_quarantine`) this call is nested
    # inside; this one only needs the notes that DID parse.
    parsed_notes, _issues, _degraded, _sources = _load_notes(store)
    notes_by_id = {note.id: note for note in parsed_notes}
    marked = 0
    for report in list(store.iter_reports()):
        members = membership.get(report.lineage_id)
        if members is None:
            continue
        expected = community_input_hash(members, tables, notes_by_id)
        if report.input_hash != expected and not report.stale:
            report.stale = True
            report.input_hash = expected
            try:
                store.write_report(report)
            except ConflictError:
                # Someone edited this report between our read and our
                # write; leave it for the next pass rather than crashing
                # every other report's staleness update over one race.
                continue
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
    *,
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
    kept: Iterable[tuple[str, str]] = (),
) -> RebuildReport:
    synced = 0
    if scope in {"cache", "all"}:
        synced = sync(
            conn, store, config, embedder, statuses,
            adoptions=adoptions, merges=merges, kept=kept,
        ).notes

    # Quarantine-aware, not a bare `fold()` call: a note with a typo'd edge
    # type or a duplicate assertion id must not take this whole call down --
    # `write_note`, `resolve_assertion`, `cluster`, and every other caller of
    # `rebuild` route through here (spec §10). See
    # `mindpalace.index.sync.fold_notes_with_quarantine`.
    notes, notes_by_id, tables, fold_issues = fold_notes_with_quarantine(
        store, config, statuses, adoptions, merges
    )
    # Persist what quarantine found. When scope includes "cache", `sync`
    # (above) already wrote the identical set as part of its own full
    # rewrite of `vault_issues`; this is a kind-scoped replace, so it is a
    # no-op in that case rather than a duplicate. When scope is
    # "related_blocks" only, `sync` never ran this call, and without this,
    # a note `fold` rejects would be quarantined correctly but invisibly --
    # `review_queue` would never learn about it.
    with conn:
        conn.execute(
            f"DELETE FROM vault_issues WHERE kind IN "
            f"({','.join('?' for _ in _FOLD_ISSUE_KINDS)})",
            _FOLD_ISSUE_KINDS,
        )
        conn.executemany(
            "INSERT INTO vault_issues (path, kind, detail) VALUES (?, ?, ?)",
            fold_issues,
        )

    written = 0
    marked_stale = 0
    if scope in {"related_blocks", "all"}:
        for entity in tables.entities.values():
            expected = entity_input_hash(entity, tables, notes_by_id)
            try:
                existing = store.read_entity_page(entity.slug)
            except FrontMatterError:
                # The file exists but cannot be parsed: leave it alone
                # (spec §10 -- Tier 2 is never deleted or clobbered by
                # rebuild) rather than let it take this whole call down.
                # `sync`'s own pass already recorded it as a vault_issue.
                continue
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

            new_type = existing.user.get("type", entity.type)
            new_generated_from = list(entity.note_ids)
            new_stale = existing.stale or became_stale
            unchanged = (
                existing.type == new_type
                and existing.generated_from == new_generated_from
                and existing.input_hash == expected
                and existing.stale == new_stale
                and existing.related == related
            )
            if unchanged:
                # Nothing to write: rewriting anyway would still be
                # idempotent, but at the cost of a Tier-2 file touch (and a
                # git diff) on every single rebuild -- and rebuild runs on
                # every write tool. See spec §9.1/§9.2: unconditional
                # rewrite here also unconditionally widens the CAS window
                # below on content that hasn't moved.
                continue

            try:
                store.write_entity_page(
                    EntityPage(
                        slug=entity.slug,
                        type=new_type,
                        description=existing.description,
                        generated_from=new_generated_from,
                        input_hash=expected,
                        stale=new_stale,
                        user=existing.user,
                        related=related,
                        trailing=existing.trailing,
                        source_hash=existing.source_hash,
                    )
                )
            except ConflictError:
                # Someone (Obsidian, another tool call) wrote this file
                # between our read and our write. Leave it for the next
                # rebuild rather than crashing this one over one page.
                continue
            written += 1

    reports_marked = mark_stale_reports(conn, store, tables)

    # Entity pages are written *after* the first sync, so their docs and vectors
    # would otherwise lag a full cycle behind. Re-sync once they exist.
    if written or reports_marked:
        sync(
            conn, store, config, embedder, statuses,
            adoptions=adoptions, merges=merges, kept=kept,
        )

    return RebuildReport(
        synced=synced,
        pages_written=written,
        pages_marked_stale=marked_stale,
        reports_marked_stale=reports_marked,
    )
