"""Record types for every artifact the vault stores."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Decision:
    op: str
    ts: str
    assertion: str
    action: str
    via: str
    reason: str | None = None


def capture_to_markdown(capture: Capture) -> str:
    data = {"id": capture.id, "created": capture.created, "source": capture.source}
    if capture.why is not None:
        data["why"] = capture.why
    return render(data, capture.text)


def capture_from_markdown(text: str) -> Capture:
    data, body = parse(text)
    return Capture(
        id=data["id"],
        created=data["created"],
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
        created=data["created"],
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
