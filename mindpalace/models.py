"""Record types for every artifact the vault stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from mindpalace.frontmatter import parse, render


@dataclass(frozen=True)
class EntityInstance:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class RelationshipAssertion:
    id: str
    source: str
    target: str
    type: str
    strength: int
    description: str


@dataclass(frozen=True)
class ClaimAssertion:
    id: str
    subject: str
    text: str


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
        data["entities"] = [
            {"name": e.name, "type": e.type, "description": e.description}
            for e in note.entities
        ]
    if note.relationship_assertions:
        data["relationship_assertions"] = [
            {
                "id": r.id,
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "strength": r.strength,
                "description": r.description,
            }
            for r in note.relationship_assertions
        ]
    if note.claim_assertions:
        data["claim_assertions"] = [
            {"id": c.id, "subject": c.subject, "text": c.text}
            for c in note.claim_assertions
        ]
    return render(data, note.body)


def note_from_markdown(text: str) -> Note:
    data, body = parse(text)
    return Note(
        id=data["id"],
        derived_from=data.get("derived_from"),
        created=_timestamp(data["created"], data.get("id")),
        author=data["author"],
        body=body.rstrip("\n"),
        entities=tuple(
            EntityInstance(name=e["name"], type=e["type"], description=e["description"])
            for e in data.get("entities", [])
        ),
        relationship_assertions=tuple(
            RelationshipAssertion(
                id=r["id"],
                source=r["source"],
                target=r["target"],
                type=r["type"],
                strength=r["strength"],
                description=r["description"],
            )
            for r in data.get("relationship_assertions", [])
        ),
        claim_assertions=tuple(
            ClaimAssertion(id=c["id"], subject=c["subject"], text=c["text"])
            for c in data.get("claim_assertions", [])
        ),
    )
