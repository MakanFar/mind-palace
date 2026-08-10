"""Derive the graph from the complete current source set.

This module is a pure function. Given every note plus the decision log it
produces entities, aggregate relationships, assertions, and claims. It has no
incremental path by design: recomputation from scratch is what makes
re-processing an edited or re-detected file safe.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from mindpalace.config import Config
from mindpalace.ids import aggregate_key, slugify
from mindpalace.models import Note

ACTION_TO_STATUS = {"confirm": "confirmed", "dismiss": "dismissed"}
UNKNOWN_TYPE = "unknown"


@dataclass(frozen=True)
class FoldedEntity:
    slug: str
    type: str
    rank: int
    note_ids: tuple[str, ...]


@dataclass(frozen=True)
class FoldedAssertion:
    id: str
    note_id: str
    source: str
    target: str
    type: str
    strength: int
    description: str
    status: str


@dataclass(frozen=True)
class FoldedClaim:
    id: str
    note_id: str
    subject: str
    text: str
    status: str


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
        raise ValueError(
            f"assertion {item_id!r}: unrecognised decision action {action!r} "
            f"(expected one of {sorted(ACTION_TO_STATUS)})"
        ) from None


def fold(
    notes: Iterable[Note], statuses: dict[str, str], config: Config
) -> GraphTables:
    ordered = sorted(notes, key=lambda note: (note.created, note.id))

    entity_types: dict[str, str] = {}
    entity_notes: dict[str, list[str]] = {}
    assertions: dict[str, FoldedAssertion] = {}
    claims: dict[str, FoldedClaim] = {}
    grouped: dict[str, list[FoldedAssertion]] = {}
    aggregate_shape: dict[str, tuple[str, str, str]] = {}

    def touch(slug: str, note_id: str, declared_type: str | None) -> None:
        if declared_type is not None or slug not in entity_types:
            entity_types[slug] = declared_type or UNKNOWN_TYPE
        note_ids = entity_notes.setdefault(slug, [])
        if note_id not in note_ids:
            note_ids.append(note_id)

    for note in ordered:
        for instance in note.entities:
            touch(slugify(instance.name), note.id, instance.type)

        for raw in note.relationship_assertions:
            if raw.type not in config.edge_types:
                raise ValueError(
                    f"note {note.id}: unknown edge type {raw.type!r} "
                    f"(assertion {raw.id})"
                )
            if raw.id in assertions:
                raise ValueError(
                    f"duplicate relationship assertion id {raw.id!r}: already "
                    f"recorded from note {assertions[raw.id].note_id!r}, seen "
                    f"again in note {note.id!r}. Assertion ids must be unique "
                    f"across the source set -- re-processing a note must "
                    f"produce fresh ids, never reuse one, or aggregates would "
                    f"silently double-count."
                )
            source, target = slugify(raw.source), slugify(raw.target)
            touch(source, note.id, None)
            touch(target, note.id, None)

            folded = FoldedAssertion(
                id=raw.id,
                note_id=note.id,
                source=source,
                target=target,
                type=raw.type,
                strength=raw.strength,
                description=raw.description,
                status=_status(raw.id, statuses),
            )
            assertions[raw.id] = folded

            key = aggregate_key(
                source, raw.type, target, symmetric=config.is_symmetric(raw.type)
            )
            grouped.setdefault(key, []).append(folded)
            if key not in aggregate_shape:
                left, _, rest = key.removeprefix("r:").partition("|")
                edge_type, _, right = rest.partition("|")
                aggregate_shape[key] = (left, edge_type, right)

        for raw_claim in note.claim_assertions:
            if raw_claim.id in claims:
                raise ValueError(
                    f"duplicate claim assertion id {raw_claim.id!r}: already "
                    f"recorded from note {claims[raw_claim.id].note_id!r}, "
                    f"seen again in note {note.id!r}. Assertion ids must be "
                    f"unique across the source set."
                )
            subject = slugify(raw_claim.subject)
            touch(subject, note.id, None)
            claims[raw_claim.id] = FoldedClaim(
                id=raw_claim.id,
                note_id=note.id,
                subject=subject,
                text=raw_claim.text,
                status=_status(raw_claim.id, statuses),
            )

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
        )
        for slug in sorted(entity_types)
    }

    return GraphTables(
        entities=entities,
        aggregates=aggregates,
        assertions=assertions,
        claims=claims,
    )
