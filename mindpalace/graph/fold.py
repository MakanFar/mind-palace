"""Derive the graph from the complete current source set.

This module is a pure function. Given every note plus the decision log it
produces entities, aggregate relationships, assertions, and claims. It has no
incremental path by design: recomputation from scratch is what makes
re-processing an edited or re-detected file safe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from mindpalace.config import Config
from mindpalace.ids import aggregate_key, slugify
from mindpalace.models import Note

ACTION_TO_STATUS = {"confirm": "confirmed", "dismiss": "dismissed"}
UNKNOWN_TYPE = "unknown"
#: A claim a *confirmed* later claim names in `supersedes`
#: (docs/decisions/0001 §4). Never a decision-log action: it is derived.
SUPERSEDED = "superseded"


class FoldError(ValueError):
    """Base for every condition `fold` refuses to silently paper over.

    A caller processing a whole vault (see the `sync` pipeline, which folds
    every note unconditionally and must keep the vault usable when
    individual files are malformed) can catch this one class to know "the
    source set failed validation," or catch a specific subtype below to
    read the structured attributes it needs to quarantine the offending
    note(s) or assertion and continue rebuilding everything else.
    """


class DuplicateAssertionIdError(FoldError):
    """An assertion or claim id was seen in more than one note.

    Assertion ids are meant to be unique across the whole current source
    set; reusing one is a caller bug (e.g. failing to mint a fresh id when
    a note is regenerated). Left unchecked, it doesn't just drop a record
    -- it can double-count into an aggregate's weight while the published
    `assertions`/`claims` table silently keeps only the last one seen.
    """

    def __init__(self, assertion_id: str, note_ids: tuple[str, ...]):
        self.assertion_id = assertion_id
        self.note_ids = note_ids
        first, second = note_ids
        super().__init__(
            f"duplicate assertion id {assertion_id!r}: already recorded from "
            f"note {first!r}, seen again in note {second!r}. Assertion ids "
            f"must be unique across the source set -- re-processing a note "
            f"must produce fresh ids, never reuse one, or aggregates would "
            f"silently double-count."
        )


class UnknownEdgeTypeError(FoldError):
    """A relationship assertion names an edge type absent from config."""

    def __init__(self, edge_type: str, note_id: str, assertion_id: str):
        self.edge_type = edge_type
        self.note_id = note_id
        self.assertion_id = assertion_id
        super().__init__(
            f"note {note_id}: unknown edge type {edge_type!r} "
            f"(assertion {assertion_id})"
        )


class UnknownDecisionActionError(FoldError):
    """The decision log carries an action `fold` doesn't recognise.

    `DecisionLog.append` now rejects unknown actions at write time, so this
    should be unreachable in practice; it remains here as defence in depth
    against a decision log written by an older or buggy caller.
    """

    def __init__(self, action: str, assertion_id: str):
        self.action = action
        self.assertion_id = assertion_id
        super().__init__(
            f"assertion {assertion_id!r}: unrecognised decision action "
            f"{action!r} (expected one of {sorted(ACTION_TO_STATUS)})"
        )


class MergeCycleError(FoldError):
    """The merge map sends a slug back to itself (a -> b -> a).

    `merge_entities` refuses to write one, so this is defence against a
    hand-edited `merges.jsonl`; the quarantine loop drops the offending entry
    and records an issue rather than failing the whole rebuild.
    """

    def __init__(self, slug: str):
        self.slug = slug
        super().__init__(
            f"merge cycle through {slug!r}: following `merges.jsonl` from it "
            f"never reaches a canonical entity"
        )


@dataclass(frozen=True)
class FoldedEntity:
    slug: str
    type: str
    rank: int
    note_ids: tuple[str, ...]
    # Slugs folded into this one by `merges.jsonl` (docs/decisions/0001 §5).
    merged_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class FoldedAssertion:
    id: str
    note_id: str
    source: str
    target: str
    # None: untyped. The model's wording is in `proposed_type` and no
    # aggregate is formed (docs/decisions/0001 §1).
    type: str | None
    strength: int
    description: str
    status: str
    proposed_type: str | None = None
    direction_corrected: bool = False


@dataclass(frozen=True)
class FoldedClaim:
    id: str
    note_id: str
    subject: str
    text: str
    status: str
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes: str | None = None


@dataclass(frozen=True)
class Aggregate:
    key: str
    source: str
    type: str
    target: str
    weight: int
    mean_strength: float
    assertion_ids: tuple[str, ...]
    traversable: bool


@dataclass(frozen=True)
class GraphTables:
    entities: dict[str, FoldedEntity] = field(default_factory=dict)
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    assertions: dict[str, FoldedAssertion] = field(default_factory=dict)
    claims: dict[str, FoldedClaim] = field(default_factory=dict)


def _status(item_id: str, statuses: dict[str, str]) -> str:
    """Resolve an id's status from the decision log's last-write-wins map.

    An id absent from `statuses` has never been reviewed: "proposed". An id
    present with an action we don't recognise is a data problem upstream
    (a typo'd action, a schema drift) and must be surfaced, not silently
    treated as "proposed" -- that would hide a real decision.
    """
    if item_id not in statuses:
        return "proposed"
    action = statuses[item_id]
    try:
        return ACTION_TO_STATUS[action]
    except KeyError:
        raise UnknownDecisionActionError(action=action, assertion_id=item_id) from None


def resolve_merges(merges: Mapping[str, str] | None) -> dict[str, str]:
    """Flatten a one-hop merge map into duplicate -> final canonical.

    Chains (a -> b, b -> c) are followed to their end. A cycle raises
    `MergeCycleError` naming a slug on it.
    """
    flat: dict[str, str] = {}
    for start in merges or {}:
        seen = [start]
        cursor = start
        while cursor in merges:
            cursor = merges[cursor]
            if cursor in seen:
                raise MergeCycleError(slug=cursor)
            seen.append(cursor)
        flat[start] = cursor
    return flat


def _adopted(adoptions: Mapping[str, Mapping[str, str]] | None, kind: str, proposed: str | None) -> str | None:
    if proposed is None or not adoptions:
        return None
    return adoptions.get(kind, {}).get(slugify(proposed))


def fold(
    notes: Iterable[Note],
    statuses: dict[str, str],
    config: Config,
    *,
    adoptions: Mapping[str, Mapping[str, str]] | None = None,
    merges: Mapping[str, str] | None = None,
) -> GraphTables:
    """`adoptions` is `VocabularyLog.adoptions()`; `merges` is
    `MergeLog.merges()`. Both default to empty so callers that predate them
    (and every test that folds a bare note list) keep working."""
    ordered = sorted(notes, key=lambda note: (note.created, note.id))
    canonical_of = resolve_merges(merges)

    def canon(slug: str) -> str:
        return canonical_of.get(slug, slug)

    entity_types: dict[str, str] = {}
    entity_notes: dict[str, list[str]] = {}
    merged_from: dict[str, set[str]] = {}
    assertions: dict[str, FoldedAssertion] = {}
    claims: dict[str, FoldedClaim] = {}
    grouped: dict[str, list[FoldedAssertion]] = {}
    aggregate_shape: dict[str, tuple[str, str, str]] = {}

    def touch(raw_slug: str, note_id: str, declared_type: str | None) -> None:
        slug = canon(raw_slug)
        if slug != raw_slug:
            merged_from.setdefault(slug, set()).add(raw_slug)
        if declared_type is not None or slug not in entity_types:
            entity_types[slug] = declared_type or UNKNOWN_TYPE
        note_ids = entity_notes.setdefault(slug, [])
        if note_id not in note_ids:
            note_ids.append(note_id)

    for note in ordered:
        for instance in note.entities:
            declared = instance.type
            if declared is None:
                declared = _adopted(adoptions, "entity", instance.proposed_type)
                if declared is not None and declared not in config.entity_types:
                    declared = None
            touch(slugify(instance.name), note.id, declared)

        for raw in note.relationship_assertions:
            if raw.type is not None and raw.type not in config.edge_types:
                raise UnknownEdgeTypeError(
                    edge_type=raw.type, note_id=note.id, assertion_id=raw.id
                )
            if raw.id in assertions:
                raise DuplicateAssertionIdError(
                    assertion_id=raw.id,
                    note_ids=(assertions[raw.id].note_id, note.id),
                )
            source, target = canon(slugify(raw.source)), canon(slugify(raw.target))
            touch(slugify(raw.source), note.id, None)
            touch(slugify(raw.target), note.id, None)

            # An adoption may name a type that has since left the config.
            # That is not the note's fault: the proposal simply stands.
            edge_type = raw.type
            if edge_type is None:
                adopted = _adopted(adoptions, "edge", raw.proposed_type)
                if adopted in config.edge_types:
                    edge_type = adopted

            folded = FoldedAssertion(
                id=raw.id,
                note_id=note.id,
                source=source,
                target=target,
                type=edge_type,
                strength=raw.strength,
                description=raw.description,
                status=_status(raw.id, statuses),
                proposed_type=raw.proposed_type,
                direction_corrected=raw.direction_corrected,
            )
            assertions[raw.id] = folded
            if edge_type is None:
                # Untyped: nothing to group by. The endpoints and the review
                # status are kept; no aggregate, no rank (0001 §1).
                continue

            symmetric = config.is_symmetric(edge_type)
            key = aggregate_key(source, edge_type, target, symmetric=symmetric)
            grouped.setdefault(key, []).append(folded)
            if key not in aggregate_shape:
                # Capture (source, type, target) directly from values already
                # in hand -- never re-derive them by parsing `key` apart.
                # `key`'s "|"-joined format has no character restriction on
                # edge-type names (config.load_config doesn't enforce one),
                # so an edge type containing "|" would silently mis-split.
                left, right = sorted((source, target)) if symmetric else (source, target)
                aggregate_shape[key] = (left, edge_type, right)

        for raw_claim in note.claim_assertions:
            if raw_claim.id in claims:
                raise DuplicateAssertionIdError(
                    assertion_id=raw_claim.id,
                    note_ids=(claims[raw_claim.id].note_id, note.id),
                )
            subject = canon(slugify(raw_claim.subject))
            touch(slugify(raw_claim.subject), note.id, None)
            claims[raw_claim.id] = FoldedClaim(
                id=raw_claim.id,
                note_id=note.id,
                subject=subject,
                text=raw_claim.text,
                status=_status(raw_claim.id, statuses),
                valid_from=raw_claim.valid_from,
                valid_to=raw_claim.valid_to,
                supersedes=raw_claim.supersedes,
            )

    # Supersession takes effect on confirmation only: a proposed correction
    # must not retire the claim it corrects before anyone has agreed to it.
    for claim in list(claims.values()):
        if claim.status == "confirmed" and claim.supersedes in claims:
            old = claims[claim.supersedes]
            claims[old.id] = replace(old, status=SUPERSEDED)

    aggregates: dict[str, Aggregate] = {}
    degree: dict[str, int] = {}
    for key, members in grouped.items():
        confirmed = [m for m in members if m.status == "confirmed"]
        left, edge_type, right = aggregate_shape[key]
        traversable = bool(confirmed)
        aggregates[key] = Aggregate(
            key=key,
            source=left,
            type=edge_type,
            target=right,
            weight=len(confirmed),
            mean_strength=(
                sum(m.strength for m in confirmed) / len(confirmed) if confirmed else 0.0
            ),
            assertion_ids=tuple(sorted(m.id for m in members)),
            traversable=traversable,
        )
        if traversable:
            degree[left] = degree.get(left, 0) + 1
            degree[right] = degree.get(right, 0) + 1

    entities = {
        slug: FoldedEntity(
            slug=slug,
            type=entity_types[slug],
            rank=degree.get(slug, 0),
            note_ids=tuple(entity_notes[slug]),
            merged_from=tuple(sorted(merged_from.get(slug, ()))),
        )
        for slug in sorted(entity_types)
    }

    return GraphTables(
        entities=entities,
        aggregates=aggregates,
        assertions=assertions,
        claims=claims,
    )
