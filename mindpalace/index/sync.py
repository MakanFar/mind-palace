"""Full rebuild of the Tier 3 cache from Tier 1 sources."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mindpalace.atomic import atomic_write, content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.graph.fold import (
    DuplicateAssertionIdError,
    GraphTables,
    MergeCycleError,
    UnknownDecisionActionError,
    UnknownEdgeTypeError,
    fold,
)
from mindpalace.graph.duplicates import (
    entity_profile,
    propose_near_duplicates,
    propose_similar_entities,
)
from mindpalace.graph.integrity import check_entity_types, check_signatures
from mindpalace.ids import slugify, unit_id
from mindpalace.index.graph_export import build_graph_export, first_line, write_graph_export
from mindpalace.index import db, vectors
from mindpalace.models import (
    Capture,
    CommunityReport,
    EntityPage,
    Note,
    capture_from_markdown,
    note_from_markdown,
)
from mindpalace.oplog import now_iso
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

    Also includes `entities/` and `communities/`: they are Tier 2 (generated
    prose, not source truth), but their description/summary text is indexed
    into `docs`/`vectors` same as a note body, so an Obsidian edit to either
    -- or a hand-added `user.aliases` -- is a change this cache depends on
    exactly as much as a note edit is. Omitting them left such an edit
    invisible to `has_drift`, so `heal()` could never notice it (spec §10).
    """
    yield from sorted(store.paths.captures.glob("*.md"))
    yield from sorted(store.paths.notes.glob("*.md"))
    yield from sorted(store.paths.entities.glob("*.md"))
    yield from sorted(store.paths.communities.glob("*.md"))
    for dependency in (
        store.paths.decisions_log,
        store.paths.mindpalace_md,
        store.paths.vocabulary_log,
        store.paths.merges_log,
        store.paths.retirements_log,
    ):
        if dependency.exists():
            yield dependency


def _source_hashes(store: VaultStore) -> Iterator[tuple[str, str]]:
    """`(relative path, content hash)` for every file in `_iter_source_files`:
    what `sync` records in `files` and what `has_drift` compares against it.
    One definition, so the two can never disagree on path or encoding."""
    for path in _iter_source_files(store):
        yield _relative(store, path), content_hash(path.read_text(encoding="utf-8"))


#: `sync` takes the fold's decision statuses either as the folded map or as
#: a callable that reads it. The callable form exists for one reason: the
#: Obsidian window appends to decisions.jsonl without the vault lock, and a
#: line landing after the map was read but before the file was hashed would
#: be recorded as folded without ever being folded (docs/decisions/0003
#: §Part 2). Deferring the read until after the hash closes that window.
StatusSource = Mapping[str, str] | Callable[[], Mapping[str, str]]


def resolve_statuses(statuses: StatusSource) -> dict[str, str]:
    return dict(statuses() if callable(statuses) else statuses)


def _load_notes(
    store: VaultStore,
) -> tuple[
    list[Note],
    list[tuple[str, str, str]],
    list[tuple[str, str]],
    dict[str, tuple[str, str]],
]:
    """Parse every note file, degrading rather than raising on failure.

    Returns successfully-parsed notes, issues recorded so far, (path,
    raw_text) pairs for files that failed to parse at all, and a note_id ->
    (path, raw_text) map for every note that DID parse -- kept around so
    that a note `fold` later rejects (see `_fold_with_quarantine`) can still
    be indexed as a degraded document by its path, without re-reading the
    file from disk.
    """
    notes: list[Note] = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []
    note_sources: dict[str, tuple[str, str]] = {}

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

    return notes, issues, degraded, note_sources


def _load_sources(
    store: VaultStore,
) -> tuple[
    list[Capture],
    list[Note],
    list[tuple[str, str, str]],
    list[tuple[str, str]],
    dict[str, tuple[str, str]],
    dict[str, str],
]:
    """Parse every Tier 1 source file, degrading rather than dropping on
    failure. See `_load_notes` for the notes half of this. The last item
    maps capture id -> relative path, for the graph export."""
    captures: list[Capture] = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []

    seen_ids: dict[str, str] = {}
    for path in sorted(store.paths.captures.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            capture = capture_from_markdown(raw)
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_capture", str(exc)))
            degraded.append((_relative(store, path), raw))
            continue
        # A copied file (Obsidian, Dropbox) carries the same id. Unit ids
        # derive from the capture id, so admitting both would collide in
        # `text_units`; keep the first, report the rest, keep it searchable.
        if capture.id in seen_ids:
            issues.append(
                (
                    _relative(store, path),
                    "duplicate_capture_id",
                    f"capture id {capture.id!r} is already used by {seen_ids[capture.id]}",
                )
            )
            degraded.append((_relative(store, path), raw))
            continue
        seen_ids[capture.id] = _relative(store, path)
        captures.append(capture)

    notes, note_issues, note_degraded, note_sources = _load_notes(store)
    issues.extend(note_issues)
    degraded.extend(note_degraded)

    return captures, notes, issues, degraded, note_sources, seen_ids


def _load_entity_pages(
    store: VaultStore,
) -> tuple[list, list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Parse every entity page, degrading rather than raising on failure --
    the Tier 2 counterpart of `_load_notes` (spec §10: one bad file must
    never render the vault unusable, and that applies to hand-edited Tier 2
    exactly as it does to Tier 1). `VaultStore.iter_entity_pages` already
    skips a page it cannot parse; this wraps the same read but also records
    *which* file was skipped, as a `vault_issue`, and keeps its raw text
    around so it can still be indexed as an `unparsed` document.
    """
    pages = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []
    for path in sorted(store.paths.entities.glob("*.md")):
        try:
            page = store.read_entity_page(path.stem)
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_entity_page", str(exc)))
            degraded.append((_relative(store, path), path.read_text(encoding="utf-8")))
            continue
        if page is not None:
            pages.append(page)
    return pages, issues, degraded


def _load_reports(
    store: VaultStore,
) -> tuple[list, list[tuple[str, str, str]], list[tuple[str, str]]]:
    """The community-report counterpart of `_load_entity_pages`."""
    reports = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []
    for path in sorted(store.paths.communities.glob("*.md")):
        try:
            report = store.read_report(path.stem)
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_report", str(exc)))
            degraded.append((_relative(store, path), path.read_text(encoding="utf-8")))
            continue
        if report is not None:
            reports.append(report)
    return reports, issues, degraded


def _fold_with_quarantine(
    notes: list[Note],
    statuses: dict[str, str],
    config: Config,
    issues: list[tuple[str, str, str]],
    note_paths: dict[str, str],
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
    retired: Iterable[str] = (),
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
    merge_map = dict(merges or {})
    excluded: set[str] = set()

    # Each accepted iteration strictly shrinks len(candidates) + len(status_map)
    # + len(merge_map) by at least one, so this many attempts is always enough
    # for a set of exceptions that shrink correctly. A future exception type
    # (or a bug) that doesn't shrink on would otherwise spin forever; bound it.
    max_iterations = len(notes) + len(statuses) + len(merge_map) + 1

    for _ in range(max_iterations):
        try:
            return (
                fold(
                    candidates.values(),
                    status_map,
                    config,
                    adoptions=adoptions,
                    merges=merge_map,
                    retired=retired,
                ),
                excluded,
            )
        except MergeCycleError as exc:
            issues.append(
                (
                    ".mindpalace/merges.jsonl",
                    "merge_cycle",
                    f"merging {exc.slug!r} follows a cycle back to itself; "
                    f"that entry is ignored until an `unmerge` breaks the loop",
                )
            )
            merge_map.pop(exc.slug, None)
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


def fold_notes_with_quarantine(
    store: VaultStore,
    config: Config,
    statuses: dict[str, str],
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
    retired: Iterable[str] = (),
) -> tuple[list[Note], dict[str, Note], GraphTables, list[tuple[str, str, str]]]:
    """Parse and fold every note currently on disk, quarantining anything
    `fold` rejects instead of raising.

    This is the ONE shared entry point every caller that needs the folded
    graph must use -- not `fold` directly. `rebuild()` and `tools._tables()`
    both used to call `fold(notes, statuses, config)` bare, which defeats
    `sync`'s quarantine: a note with a typo'd edge type (or a duplicate
    assertion id from a git merge, or an edge type MINDPALACE.md stopped
    declaring) makes `fold` raise a typed `FoldError`, and a bare call lets
    that propagate straight out of `write_note`, `propose_relationship`,
    `resolve_assertion`, `rebuild`, `cluster`, `write_entity_description`,
    and `write_community_report` -- the entire write surface goes dead over
    one bad note, which is exactly what spec §10 says must never happen.

    Returns the notes that survived quarantine, a note_id -> Note map of the
    same, the folded tables, and every issue recorded along the way
    (including a raw parse failure -- reused from `_load_notes` -- so a note
    that isn't even valid YAML is quarantined the same way as one `fold`
    itself rejects).
    """
    notes, issues, _degraded, note_sources = _load_notes(store)
    note_paths = {note_id: path for note_id, (path, _raw) in note_sources.items()}
    tables, excluded = _fold_with_quarantine(
        notes, statuses, config, issues, note_paths, adoptions, merges, retired
    )
    issues.extend(_revived_issues(tables))
    folded_notes = [note for note in notes if note.id not in excluded]
    notes_by_id = {note.id: note for note in folded_notes}
    return folded_notes, notes_by_id, tables, issues


_LOCATOR_HEADING = re.compile(r"^## ((?:Page|Slide|Sheet|Section|Line) .+)$", re.MULTILINE)


@dataclass(frozen=True)
class CaptureUnit:
    id: str
    capture_id: str
    ordinal: int
    start: int
    end: int
    locator: str | None
    text: str


def capture_units(capture: Capture) -> list[CaptureUnit]:
    """The units of one capture (docs/decisions/0002 §Storage). A capture
    with no recorded offsets is one unit spanning its whole text. The
    locator is the nearest `## Page N`-style heading at or before the unit."""
    spans = capture.units or ((0, len(capture.text)),)
    units: list[CaptureUnit] = []
    for ordinal, (start, end) in enumerate(spans):
        text = capture.text[start:end]
        # The page a unit *starts* on: a heading at its very start, else
        # the last one before it. A heading further inside the unit belongs
        # to the text after it, not to the unit as a whole.
        heading = _LOCATOR_HEADING.match(text.lstrip())
        if heading is not None:
            locator = heading.group(1)
        else:
            before = _LOCATOR_HEADING.findall(capture.text[:start])
            locator = before[-1] if before else None
        units.append(
            CaptureUnit(unit_id(capture.id, ordinal), capture.id, ordinal, start, end, locator, text)
        )
    return units


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
        # Dedupe per page before recording a claim: a page listing the same
        # alias twice (or the same alias in two different cases -- both
        # normalise to the same slug) must count as one claim, not two, or
        # it self-collides and gets reported as ambiguous against itself.
        for alias in {slugify(a) for a in page.user.get("aliases", [])}:
            owners.setdefault(alias, []).append(page.slug)

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


def _near_duplicate_issues(
    store: VaultStore, tables: GraphTables, entity_pages: list
) -> list[tuple[str, str, str]]:
    """The Phase-1 near-duplicate lint (spec §10), surfaced like every other
    deferred-cleanup finding: reported through `review_queue`, never acted on.
    Spec §8.5 forbids merging identities, so the remedy is always a human
    adding one slug to the other's `user.aliases` -- which is exactly what
    `declared_aliases` then suppresses on the next sync.
    """
    declared: dict[str, set[str]] = {}
    for page in entity_pages:
        user = page.user if isinstance(page.user, dict) else {}
        declared[page.slug] = {slugify(a) for a in user.get("aliases", [])}

    issues: list[tuple[str, str, str]] = []
    for pair in propose_near_duplicates(tables, declared):
        paths = ", ".join(
            _relative(store, store.paths.entity_path(slug))
            for slug in (pair.base, pair.superset)
        )
        issues.append(
            (
                paths,
                "near_duplicate_entity",
                f"{pair.base!r} and {pair.superset!r} may be the same entity; "
                f"if they are, add one to the other's `user.aliases` -- "
                f"mindpalace does not merge identities for you",
            )
        )
    return issues


def _entity_type_issues(
    store: VaultStore, notes: list[Note], tables: GraphTables, config: Config
) -> list[tuple[str, str, str]]:
    """Entity-type integrity findings, reported and never acted on.

    `fold` validates edge types but not entity types, because an edge type
    drives real behaviour (`is_symmetric`, `cluster_weight`) while an entity
    type is only a label -- there is nothing `fold` cannot do with a wrong
    one, so refusing the note would be a harsher remedy than the problem
    warrants. That leaves the label free to be wrong with no signal at all,
    which is what these three checks supply.

    `notes` must already have quarantined notes filtered out: a note the
    graph rejected is a problem the user is being pointed at anyway, and
    re-reporting its entity types would send them to a second finding that
    vanishes the moment they fix the first.
    """
    issues: list[tuple[str, str, str]] = []
    for finding in check_entity_types(notes, tables, config):
        issues.append(
            (
                _relative(store, store.paths.entity_path(finding.slug)),
                finding.kind,
                finding.detail,
            )
        )
    return issues


def _similar_entity_issues(
    store: VaultStore,
    tables: GraphTables,
    entity_pages: list,
    profiles: Mapping[str, object],
    config: Config,
    kept: Iterable[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Stage two of the duplicate lint (docs/decisions/0001 §6). The issue
    text tells the assistant what its stage-three verdict can be."""
    declared: dict[str, set[str]] = {}
    for page in entity_pages:
        user = page.user if isinstance(page.user, dict) else {}
        declared[page.slug] = {slugify(a) for a in user.get("aliases", [])}

    issues: list[tuple[str, str, str]] = []
    for pair in propose_similar_entities(
        profiles, tables, config.thresholds.duplicate_cosine_floor, declared, kept
    ):
        paths = ", ".join(
            _relative(store, store.paths.entity_path(slug))
            for slug in (pair.left, pair.right)
        )
        issues.append(
            (
                paths,
                "similar_entity",
                f"{pair.left!r} and {pair.right!r} have near-identical profiles "
                f"(cosine {pair.similarity:.2f}); if they are one thing, call "
                f"merge_entities(duplicate, canonical); if not, call "
                f"merge_entities(..., action='keep') so this is not asked again",
            )
        )
    return issues


def _vocabulary_proposals(
    notes: list[Note],
    config: Config,
    adoptions: Mapping[str, Mapping[str, str]] | None,
) -> list[tuple[str, str, int, str, str]]:
    """(kind, normalised proposal, count, one example, comma-joined ids).

    A wording already adopted onto a configured type is no longer a
    proposal; one adopted onto a type that has since left the config is."""
    tally: dict[tuple[str, str], list] = {}
    configured = {"edge": set(config.edge_types), "entity": set(config.entity_types)}

    def record(kind: str, wording: str | None, example: str, item_id: str) -> None:
        if not wording:
            return
        key = (kind, slugify(wording))
        if not key[1]:
            return
        adopted = (adoptions or {}).get(kind, {}).get(key[1])
        if adopted in configured[kind]:
            return
        entry = tally.setdefault(key, [0, example, []])
        entry[0] += 1
        entry[2].append(item_id)

    for note in notes:
        for instance in note.entities:
            if instance.type is None:
                record("entity", instance.proposed_type, f"{instance.name} ({instance.proposed_type})", note.id)
        for assertion in note.relationship_assertions:
            if assertion.type is None:
                record(
                    "edge",
                    assertion.proposed_type,
                    f"{assertion.source} {assertion.proposed_type} {assertion.target}",
                    assertion.id,
                )
    return [
        (kind, proposed, count, example, ",".join(ids))
        for (kind, proposed), (count, example, ids) in sorted(tally.items())
    ]


def _entity_profiles(notes: list[Note], tables: GraphTables) -> dict[str, str]:
    """slug -> profile text, for every entity the graph currently holds."""
    descriptions: dict[str, list[str]] = {}
    for note in notes:
        for instance in note.entities:
            descriptions.setdefault(slugify(instance.name), []).append(instance.description)
    profiles: dict[str, str] = {}
    for slug, entity in tables.entities.items():
        texts = list(descriptions.get(slug, []))
        for merged in entity.merged_from:
            texts.extend(descriptions.get(merged, []))
        profiles[slug] = entity_profile(slug, texts)
    return profiles


def _merged_page_issues(
    store: VaultStore, tables: GraphTables, entity_pages: list
) -> tuple[list[tuple[str, str, str]], set[str]]:
    """An entity page whose slug has been merged away (docs/decisions/0001
    §5) is orphaned: rebuild never deletes Tier 2, so the file stays, but it
    must not be indexed as a live entity or read as one. Report it and hand
    back the slugs so `sync` skips them."""
    canonical = {
        merged: entity.slug
        for entity in tables.entities.values()
        for merged in entity.merged_from
    }
    issues: list[tuple[str, str, str]] = []
    orphaned: set[str] = set()
    for page in entity_pages:
        target = canonical.get(page.slug)
        if target is None:
            continue
        orphaned.add(page.slug)
        issues.append(
            (
                _relative(store, store.paths.entity_path(page.slug)),
                "merged_entity_page",
                f"{page.slug!r} was merged into {target!r}; this page is no "
                f"longer an entity. Move any prose worth keeping to "
                f"entities/{target}.md and delete this file, or unmerge",
            )
        )
    return issues, orphaned


def _retired_page_issues(
    store: VaultStore, tables: GraphTables, entity_pages: list, retired: set[str]
) -> tuple[list[tuple[str, str, str]], set[str]]:
    """An entity page for a retired slug (docs/decisions/0004): the file stays,
    like a merged-away page, but it is not a live entity and must not be
    indexed as one. A revived slug is back in the graph, so its page is not
    reported here."""
    issues: list[tuple[str, str, str]] = []
    orphaned: set[str] = set()
    for page in entity_pages:
        if page.slug not in retired or page.slug in tables.entities:
            continue
        orphaned.add(page.slug)
        issues.append(
            (
                _relative(store, store.paths.entity_path(page.slug)),
                "retired_entity_page",
                f"{page.slug!r} is retired; this page is no longer an entity. "
                f"Delete this file, or restore the entity",
            )
        )
    return issues, orphaned


def _revived_issues(tables: GraphTables) -> list[tuple[str, str, str]]:
    """A retired slug that folded anyway (docs/decisions/0004 §Part 1): a live
    assertion or claim names it, and hiding a reviewable item would be silent
    loss, so it is back in the graph and this says so."""
    return [
        (
            ".mindpalace/retirements.jsonl",
            "retired_entity_revived",
            f"{slug!r} is retired but a live assertion or claim names it, so it "
            f"is back in the graph; decide that item and retire again, or restore",
        )
        for slug in tables.revived
    ]


def _signature_issues(
    store: VaultStore, tables: GraphTables, config: Config, note_paths: dict[str, str]
) -> list[tuple[str, str, str]]:
    return [
        (
            note_paths.get(tables.assertions[f.assertion_id].note_id, ".mindpalace"),
            "signature_violation",
            f.detail,
        )
        for f in check_signatures(tables, config)
    ]


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
        lines.append(f"- `{note.id}` {note.created[:10]} — {first_line(note.body, 120)}")
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
    statuses: StatusSource,
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
    kept: Iterable[tuple[str, str]] = (),
    retired: Iterable[str] = (),
) -> SyncReport:
    """`adoptions`, `merges`, `kept`, `retired` are the folded vocabulary,
    merge and retirement logs (see `Session.overlays`). Defaulted so a
    caller that has none still syncs.
    `statuses` is the decision map or a callable that reads it; see
    `StatusSource` for why a caller should pass the callable."""
    # Hash every source file first: before the statuses are read, before the
    # fold, and before the slow embed. The Obsidian window appends to
    # decisions.jsonl without the vault lock; a line landing anywhere after
    # this point must be hashed as *not yet* folded, or `has_drift` would
    # never notice it (docs/decisions/0003 §Part 2).
    source_hashes = list(_source_hashes(store))
    statuses = resolve_statuses(statuses)
    captures, notes, issues, degraded, note_sources, capture_paths = _load_sources(store)

    note_paths = {note_id: path for note_id, (path, _raw) in note_sources.items()}
    retired = set(retired)
    tables, excluded_notes = _fold_with_quarantine(
        notes, statuses, config, issues, note_paths, adoptions, merges, retired
    )
    issues.extend(_revived_issues(tables))

    # A note the graph rejected (unknown edge type, duplicate assertion id)
    # must still be findable so the user can go fix it -- spec §10 applies
    # to graph-level rejections exactly as it does to parse failures.
    folded_notes = [note for note in notes if note.id not in excluded_notes]
    for note_id in excluded_notes:
        degraded.append(note_sources[note_id])

    documents: list[tuple[str, str, str, str]] = []
    all_units: list[CaptureUnit] = []
    for capture in captures:
        units = capture_units(capture)
        all_units.extend(units)
        if len(units) == 1:
            documents.append((capture.id, "capture", first_line(capture.text, 120), capture.text))
            continue
        # A long capture is searched by unit so a hit lands on a page or a
        # section, and its body is not indexed as well, so it never comes
        # back twice for one query.
        for unit in units:
            documents.append(
                (unit.id, "unit", unit.locator or first_line(capture.text, 120), unit.text)
            )
    for note in folded_notes:
        # Assertion descriptions ride along in the note's searchable text rather
        # than becoming their own documents: they must be findable (spec §8.1)
        # but an `x_` id is not something `read` can return.
        assertion_text = " ".join(
            a.description for a in note.relationship_assertions
        )
        claim_text = " ".join(c.text for c in note.claim_assertions)
        searchable = "\n".join(filter(None, [note.body, assertion_text, claim_text]))
        documents.append((note.id, "note", first_line(note.body, 120), searchable))
    # Tier 2 gets the same degrade-and-record treatment as Tier 1 notes: a
    # page that fails to parse is skipped, recorded as a vault_issue, and
    # kept searchable via `degraded` rather than taking `sync` down (spec
    # §10 applies to a hand-edited entity page exactly as it does to a
    # hand-edited note).
    entity_pages, entity_issues, entity_degraded = _load_entity_pages(store)
    issues.extend(entity_issues)
    degraded.extend(entity_degraded)
    merged_issues, orphaned = _merged_page_issues(store, tables, entity_pages)
    issues.extend(merged_issues)
    retired_issues, retired_pages = _retired_page_issues(store, tables, entity_pages, retired)
    issues.extend(retired_issues)
    orphaned |= retired_pages
    for page in entity_pages:
        if page.slug in orphaned:
            continue
        documents.append((f"e_{page.slug}", "entity", page.slug, page.description))
    issues.extend(_ambiguous_alias_issues(store, entity_pages))
    issues.extend(_near_duplicate_issues(store, tables, entity_pages))
    issues.extend(_entity_type_issues(store, folded_notes, tables, config))
    issues.extend(_signature_issues(store, tables, config, note_paths))

    reports, report_issues, report_degraded = _load_reports(store)
    issues.extend(report_issues)
    degraded.extend(report_degraded)
    for report in reports:
        documents.append(
            (report.lineage_id, "report", report.title, f"{report.summary}\n{report.findings}")
        )
    # A file we cannot parse -- or a note the graph rejected -- is still
    # findable, not vanished. Keyed by path, since it has no usable id.
    for relative_path, raw in degraded:
        documents.append((relative_path, "unparsed", relative_path, raw))

    # Embed before opening the transaction. The model call is the slow part and
    # must not hold a write transaction open across it. Entity profiles for
    # the similarity lint ride in the same batch: one model call, not two.
    profile_texts = _entity_profiles(folded_notes, tables)
    profile_slugs = sorted(profile_texts)
    batch = [text for _, _, _, text in documents] + [profile_texts[s] for s in profile_slugs]
    embedded = embedder.embed(batch) if batch else []
    matrix = embedded[: len(documents)]
    profiles = dict(zip(profile_slugs, embedded[len(documents) :], strict=True))
    issues.extend(
        _similar_entity_issues(store, tables, entity_pages, profiles, config, kept)
    )
    drops = [
        (note.id, drop.kind, drop.reason, drop.detail, drop.example)
        for note in folded_notes
        for drop in note.drops
    ]
    proposals = _vocabulary_proposals(folded_notes, config, adoptions)

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
            "drops",
            "vocabulary_proposals",
            "text_units",
            "provenance",
        ):
            conn.execute(f"DELETE FROM {table}")
        vectors.clear(conn)
        db.write_meta(conn, embedder.model_id, embedder.dim)

        conn.executemany("INSERT INTO files (path, hash) VALUES (?, ?)", source_hashes)

        for entity in tables.entities.values():
            conn.execute(
                "INSERT INTO entities (slug, type, rank, merged_from) VALUES (?, ?, ?, ?)",
                (entity.slug, entity.type, entity.rank, ",".join(entity.merged_from)),
            )
            conn.executemany(
                "INSERT INTO entity_sources (slug, note_id) VALUES (?, ?)",
                [(entity.slug, note_id) for note_id in entity.note_ids],
            )

        for assertion in tables.assertions.values():
            conn.execute(
                "INSERT INTO assertions (id, kind, note_id, source, target, type, "
                "strength, description, status, proposed_type, direction_corrected) "
                "VALUES (?, 'relationship', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    assertion.id,
                    assertion.note_id,
                    assertion.source,
                    assertion.target,
                    assertion.type,
                    assertion.strength,
                    assertion.description,
                    assertion.status,
                    assertion.proposed_type,
                    int(assertion.direction_corrected),
                ),
            )

        for claim in tables.claims.values():
            conn.execute(
                "INSERT INTO claims (id, note_id, subject, text, status, "
                "valid_from, valid_to, supersedes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.id,
                    claim.note_id,
                    claim.subject,
                    claim.text,
                    claim.status,
                    claim.valid_from,
                    claim.valid_to,
                    claim.supersedes,
                ),
            )

        provenance: list[tuple[str, str]] = []
        for entity in tables.entities.values():
            provenance += [(f"e_{entity.slug}", u) for u in entity.text_unit_ids]
        for assertion in tables.assertions.values():
            provenance += [(assertion.id, u) for u in assertion.text_unit_ids]
        for claim in tables.claims.values():
            provenance += [(claim.id, u) for u in claim.text_unit_ids]
        conn.executemany(
            "INSERT OR IGNORE INTO provenance (item_id, unit_id) VALUES (?, ?)", provenance
        )
        conn.executemany(
            "INSERT INTO text_units (id, capture_id, ordinal, start, end, locator, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(u.id, u.capture_id, u.ordinal, u.start, u.end, u.locator, u.text) for u in all_units],
        )
        conn.executemany(
            "INSERT INTO drops (note_id, kind, reason, detail, example) "
            "VALUES (?, ?, ?, ?, ?)",
            drops,
        )
        conn.executemany(
            "INSERT INTO vocabulary_proposals (kind, proposed, count, example, ids) "
            "VALUES (?, ?, ?, ?, ?)",
            proposals,
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

    _write_graph_json(
        conn, store, config, tables, captures, capture_paths, folded_notes, note_paths,
        all_units, proposals, entity_pages, reports,
    )

    return SyncReport(
        notes=len(folded_notes),
        captures=len(captures),
        entities=len(tables.entities),
        aggregates=len(tables.aggregates),
        issues=tuple(issues),
    )


def _write_graph_json(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    tables: GraphTables,
    captures: list[Capture],
    capture_paths: dict[str, str],
    folded_notes: list[Note],
    note_paths: dict[str, str],
    all_units: list[CaptureUnit],
    proposals: list[tuple[str, str, int, str, str]],
    entity_pages: list[EntityPage],
    reports: list[CommunityReport],
) -> None:
    """The window's data (docs/decisions/0003). Communities are read back
    because `cluster` owns that table; everything else is in hand."""
    write_graph_export(
        store.paths.graph_json,
        build_graph_export(
            tables,
            config,
            captures,
            capture_paths,
            {note.id: note for note in folded_notes},
            note_paths,
            all_units,
            db.read_communities(conn),
            proposals,
            entity_pages,
            reports,
            now_iso(),
        ),
    )


def export_graph(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    statuses: StatusSource,
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
    retired: Iterable[str] = (),
) -> None:
    """Rewrite graph.json from the sources and the tables as they stand,
    without touching the cache or the embedder.

    For a writer that changed something the window shows but the cache does
    not derive from the sources -- `cluster` owns the communities table --
    so it can refresh the export at the cost of a parse and a fold rather
    than a full `sync` (re-embedding every document to copy one list)."""
    captures, notes, issues, _degraded, note_sources, capture_paths = _load_sources(store)
    note_paths = {note_id: path for note_id, (path, _raw) in note_sources.items()}
    tables, excluded_notes = _fold_with_quarantine(
        notes, resolve_statuses(statuses), config, issues, note_paths, adoptions, merges, retired
    )
    folded_notes = [note for note in notes if note.id not in excluded_notes]
    all_units = [unit for capture in captures for unit in capture_units(capture)]
    entity_pages, _issues, _degraded = _load_entity_pages(store)
    reports, _issues, _degraded = _load_reports(store)
    _write_graph_json(
        conn, store, config, tables, captures, capture_paths, folded_notes, note_paths,
        all_units, _vocabulary_proposals(folded_notes, config, adoptions), entity_pages, reports,
    )


def has_drift(conn: sqlite3.Connection, store: VaultStore) -> bool:
    recorded = {
        row["path"]: row["hash"] for row in conn.execute("SELECT path, hash FROM files")
    }
    seen: set[str] = set()
    for relative, current in _source_hashes(store):
        seen.add(relative)
        if recorded.get(relative) != current:
            return True
    return seen != recorded.keys()
