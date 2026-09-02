"""Record types for every artifact the vault stores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from mindpalace.frontmatter import parse, render


#: `valid_from` / `valid_to` on a claim: the string carries its own precision
#: (docs/decisions/0001 §4), so there is no separate precision field to keep
#: in step with it. `valid_to` may also be the literal "unknown": ended, date
#: unknown -- which a single nullable value cannot say.
VALIDITY_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
VALIDITY_UNKNOWN = "unknown"
_PRECISIONS = {4: "year", 7: "month", 10: "day"}


def validity_precision(value: str | None) -> str | None:
    """'year' | 'month' | 'day' | 'unknown' | None, read off the shape."""
    if value is None:
        return None
    if value == VALIDITY_UNKNOWN:
        return VALIDITY_UNKNOWN
    return _PRECISIONS.get(len(value))


def is_valid_validity(value: str | None, *, allow_unknown: bool) -> bool:
    if value is None:
        return True
    if value == VALIDITY_UNKNOWN:
        return allow_unknown
    return bool(VALIDITY_DATE.match(value))


@dataclass(frozen=True)
class EntityInstance:
    """`type` is None when the model's wording matched nothing configured;
    the wording itself is kept in `proposed_type` (docs/decisions/0001 §1)."""

    name: str
    type: str | None
    description: str
    proposed_type: str | None = None


@dataclass(frozen=True)
class RelationshipAssertion:
    id: str
    source: str
    target: str
    type: str | None
    strength: int
    description: str
    proposed_type: str | None = None
    # Endpoints were swapped at write time to satisfy the edge type's
    # domain/range (docs/decisions/0001 §3). Recorded so it is never silent.
    direction_corrected: bool = False


@dataclass(frozen=True)
class ClaimAssertion:
    id: str
    subject: str
    text: str
    valid_from: str | None = None
    valid_to: str | None = None
    # Id of the claim this one corrects. Takes effect on confirmation: the
    # older claim then folds as `superseded` (docs/decisions/0001 §4).
    supersedes: str | None = None

    @property
    def valid_from_precision(self) -> str | None:
        return validity_precision(self.valid_from)

    @property
    def valid_to_precision(self) -> str | None:
        return validity_precision(self.valid_to)


@dataclass(frozen=True)
class Drop:
    """One item `write_note` refused, and why (docs/decisions/0001 §2).

    Lives in the note's front-matter so the ledger is part of the source set:
    a cache rebuild reproduces it, and a reader of the file sees what the
    extraction lost without opening a database.
    """

    kind: str
    reason: str
    detail: str
    example: str | None = None


@dataclass(frozen=True)
class Capture:
    id: str
    created: str
    source: str
    why: str | None
    text: str


@dataclass(frozen=True)
class Note:
    id: str
    derived_from: str | None
    created: str
    author: str
    body: str
    entities: tuple[EntityInstance, ...] = ()
    relationship_assertions: tuple[RelationshipAssertion, ...] = ()
    claim_assertions: tuple[ClaimAssertion, ...] = ()
    drops: tuple[Drop, ...] = ()


@dataclass
class EntityPage:
    """Tier 2. Never deleted by rebuild; `related` is the one machine-owned block."""

    slug: str
    type: str
    description: str
    generated_from: list[str] = field(default_factory=list)
    input_hash: str = ""
    stale: bool = True
    user: dict = field(default_factory=dict)
    related: list[str] = field(default_factory=list)
    # Anything a human wrote below the `mindpalace:related` block -- durable
    # prose that `write_entity_page` must re-emit rather than discard.
    trailing: str = ""
    # The content hash of the file as it was on disk at read time, so a
    # read-then-write round trip can CAS-check before overwriting (spec
    # §9.2). Not part of equality: two pages with the same data are equal
    # regardless of which read produced them, and most callers build a
    # fresh EntityPage without ever reading one, which must not be treated
    # as "matches an empty file."
    source_hash: str | None = field(default=None, compare=False, repr=False)


@dataclass
class CommunityReport:
    """Tier 2. Keyed by stable lineage id, not by Leiden's per-run community id."""

    lineage_id: str
    level: int
    title: str
    summary: str
    rank: float
    findings: list[dict] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)
    generated_from: list[str] = field(default_factory=list)
    input_hash: str = ""
    stale: bool = False
    # See EntityPage.source_hash.
    source_hash: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Decision:
    op: str
    ts: str
    assertion: str
    action: str
    via: str
    reason: str | None = None


def _timestamp(value: object, record_id: object) -> str:
    """Normalise a front-matter `created` value to the canonical string.

    YAML parses an unquoted ISO timestamp into a `datetime` -- which is what
    a human hand-editing a note in Obsidian naturally writes. `created` is
    declared `str` and is sorted against other notes' values, so letting a
    datetime through raised `TypeError: '<' not supported between 'str' and
    'datetime.datetime'` inside `fold`. That is not a `FoldError`, so the
    quarantine pass never caught it and one hand-edited file took the whole
    vault's sync down -- exactly what spec §10 forbids.

    So: accept it and canonicalise, rather than reject. The value means the
    right thing; only its type is wrong. Anything that is not a timestamp at
    all still raises, and `_load_notes` degrades that file into a
    `malformed_note` issue the same as any other parse failure.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        # Match `tools._stamp`: UTC with a "Z", never "+00:00". These strings
        # are compared lexicographically across notes, and the two spellings
        # of the same instant do not sort equal.
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(
        f"{record_id!r}: created must be a timestamp, got "
        f"{type(value).__name__!r} ({value!r})"
    )


def capture_to_markdown(capture: Capture) -> str:
    data = {"id": capture.id, "created": capture.created, "source": capture.source}
    if capture.why is not None:
        data["why"] = capture.why
    return render(data, capture.text)


def capture_from_markdown(text: str) -> Capture:
    data, body = parse(text)
    return Capture(
        id=data["id"],
        created=_timestamp(data["created"], data.get("id")),
        source=data["source"],
        why=data.get("why"),
        text=body.rstrip("\n"),
    )


def note_to_markdown(note: Note) -> str:
    data: dict = {"id": note.id}
    if note.derived_from is not None:
        data["derived_from"] = note.derived_from
    data["created"] = note.created
    data["author"] = note.author
    if note.entities:
        data["entities"] = [_entity_to_dict(e) for e in note.entities]
    if note.relationship_assertions:
        data["relationship_assertions"] = [
            _relationship_to_dict(r) for r in note.relationship_assertions
        ]
    if note.claim_assertions:
        data["claim_assertions"] = [_claim_to_dict(c) for c in note.claim_assertions]
    if note.drops:
        data["drops"] = [_drop_to_dict(d) for d in note.drops]
    return render(data, note.body)


def _entity_to_dict(e: EntityInstance) -> dict:
    out: dict = {"name": e.name, "type": e.type, "description": e.description}
    if e.proposed_type is not None:
        out["proposed_type"] = e.proposed_type
    return out


def _relationship_to_dict(r: RelationshipAssertion) -> dict:
    out: dict = {
        "id": r.id,
        "source": r.source,
        "target": r.target,
        "type": r.type,
        "strength": r.strength,
        "description": r.description,
    }
    if r.proposed_type is not None:
        out["proposed_type"] = r.proposed_type
    # Only ever written when true: a file full of `direction_corrected: false`
    # lines would bury the one that matters.
    if r.direction_corrected:
        out["direction_corrected"] = True
    return out


def _claim_to_dict(c: ClaimAssertion) -> dict:
    out: dict = {"id": c.id, "subject": c.subject, "text": c.text}
    for key in ("valid_from", "valid_to", "supersedes"):
        value = getattr(c, key)
        if value is not None:
            out[key] = value
    return out


def _drop_to_dict(d: Drop) -> dict:
    out: dict = {"kind": d.kind, "reason": d.reason, "detail": d.detail}
    if d.example is not None:
        out["example"] = d.example
    return out


def _validity(value: object) -> str | None:
    """A bare `valid_from: 2023-05-01` is parsed by YAML as a date. Turn it
    back into the string form the precision rules are written against."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def note_from_markdown(text: str) -> Note:
    data, body = parse(text)
    return Note(
        id=data["id"],
        derived_from=data.get("derived_from"),
        created=_timestamp(data["created"], data.get("id")),
        author=data["author"],
        body=body.rstrip("\n"),
        entities=tuple(
            EntityInstance(
                name=e["name"],
                type=e.get("type"),
                description=e["description"],
                proposed_type=e.get("proposed_type"),
            )
            for e in data.get("entities", [])
        ),
        relationship_assertions=tuple(
            RelationshipAssertion(
                id=r["id"],
                source=r["source"],
                target=r["target"],
                type=r.get("type"),
                strength=r["strength"],
                description=r["description"],
                proposed_type=r.get("proposed_type"),
                direction_corrected=bool(r.get("direction_corrected", False)),
            )
            for r in data.get("relationship_assertions", [])
        ),
        claim_assertions=tuple(
            ClaimAssertion(
                id=c["id"],
                subject=c["subject"],
                text=c["text"],
                valid_from=_validity(c.get("valid_from")),
                valid_to=_validity(c.get("valid_to")),
                supersedes=c.get("supersedes"),
            )
            for c in data.get("claim_assertions", [])
        ),
        drops=tuple(
            Drop(
                kind=d["kind"],
                reason=d["reason"],
                detail=d["detail"],
                example=d.get("example"),
            )
            for d in data.get("drops", [])
        ),
    )
