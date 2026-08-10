"""Tool implementations as plain functions over a Session.

Keeping these free of MCP types makes them directly testable; server.py does
nothing but wire them to the protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mindpalace.ids import new_id, slugify
from mindpalace.models import (
    Capture,
    ClaimAssertion,
    EntityInstance,
    Note,
    RelationshipAssertion,
)
from mindpalace.rebuild import rebuild
from mindpalace.retrieve import local_search
from mindpalace.session import Session

DEFAULT_EXTRACTION_NEXT = (
    "Read any nearest notes you need, then call write_note with this capture id. "
    "Propose a relationship only where you can give a specific rationale citing "
    "both endpoints. Proposing nothing is a valid outcome."
)


class ToolError(ValueError):
    """Raised for invalid tool input. Surfaced to the assistant verbatim."""


STRENGTH_RANGE = range(1, 11)


def _require(condition: bool, message: str) -> None:
    """Every invalid input must reach the assistant as a ToolError it can act on.

    Without this, a missing dict key surfaces as a KeyError traceback and the
    assistant has no idea what to fix.
    """
    if not condition:
        raise ToolError(message)


def _require_slug(raw: str, field: str) -> str:
    """Guard against names that normalise away entirely, e.g. "!!!" -> ""."""
    slug = slugify(raw)
    _require(bool(slug), f"{field} {raw!r} normalises to an empty slug")
    return slug


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def _relative(session: Session, path) -> str:
    return str(path.relative_to(session.paths.root))


def _edge_vocabulary(session: Session) -> dict[str, str]:
    return {
        name: ("directed" if spec.directed else "symmetric")
        for name, spec in session.config.edge_types.items()
    }


def _known_entities(session: Session, limit: int = 40) -> list[dict]:
    rows = session.conn.execute(
        "SELECT slug, type, rank FROM entities ORDER BY rank DESC, slug LIMIT ?",
        (limit,),
    ).fetchall()
    entities = []
    for row in rows:
        page = session.store.read_entity_page(row["slug"])
        entities.append(
            {
                "name": row["slug"],
                "type": row["type"],
                "rank": row["rank"],
                "aliases": (page.user.get("aliases", []) if page else []),
            }
        )
    return entities


def _previously_dismissed(session: Session) -> list[dict]:
    reasons = session.decisions.dismissal_reasons()
    if not reasons:
        return []
    placeholders = ",".join("?" for _ in reasons)
    rows = session.conn.execute(
        f"SELECT id, source, type, target FROM assertions WHERE id IN ({placeholders})",
        tuple(reasons),
    ).fetchall()
    return [
        {
            "pair": f"{row['source']}|{row['type']}|{row['target']}",
            "reason": reasons[row["id"]],
        }
        for row in rows
    ]


def save_capture(
    session: Session, text: str, why: str | None = None, source: str = "manual"
) -> dict:
    if not text.strip():
        raise ToolError("capture text is empty")

    when = _now()
    capture_id = new_id("c_")
    capture = Capture(
        id=capture_id, created=_timestamp(when), source=source, why=why, text=text
    )

    with session.operation({"tool": "save_capture", "capture": capture_id}):
        path = session.store.write_capture(capture, when)
        session.resync()

    nearest = local_search(session.conn, session.embedder, text, session.config, k=5)

    return {
        "id": capture_id,
        "path": _relative(session, path),
        "nearest": nearest["hits"],
        "known_entities": _known_entities(session),
        "previously_dismissed": _previously_dismissed(session),
        "entity_types": session.config.entity_types,
        "edge_vocabulary": _edge_vocabulary(session),
        "next": session.config.templates.get(
            "extraction_next", DEFAULT_EXTRACTION_NEXT
        ),
    }


def write_note(
    session: Session,
    derived_from: str,
    content: str,
    entities: list[dict] | tuple = (),
    relationship_assertions: list[dict] | tuple = (),
    claim_assertions: list[dict] | tuple = (),
) -> dict:
    _require(bool(content.strip()), "note content is empty")

    known_captures = {capture.id for capture in session.store.iter_captures()}
    _require(derived_from in known_captures, f"no such capture: {derived_from}")

    for entity in entities:
        for field in ("name", "type", "description"):
            _require(field in entity, f"entity entry is missing {field!r}: {entity}")
        _require(
            entity["type"] in session.config.entity_types,
            f"unknown entity type {entity['type']!r}; "
            f"choose from {session.config.entity_types}",
        )
        _require_slug(entity["name"], "entity name")

    for assertion in relationship_assertions:
        for field in ("source", "target", "type", "description"):
            _require(
                field in assertion, f"relationship is missing {field!r}: {assertion}"
            )
        _require(
            assertion["type"] in session.config.edge_types,
            f"unknown edge type {assertion['type']!r}; "
            f"choose from {sorted(session.config.edge_types)}",
        )
        strength = assertion.get("strength", 5)
        _require(
            isinstance(strength, int) and strength in STRENGTH_RANGE,
            f"strength must be an integer 1-10, got {strength!r}",
        )
        _require_slug(assertion["source"], "relationship source")
        _require_slug(assertion["target"], "relationship target")

    for claim in claim_assertions:
        for field in ("subject", "text"):
            _require(field in claim, f"claim is missing {field!r}: {claim}")
        _require(bool(claim["text"].strip()), "claim text is empty")
        _require_slug(claim["subject"], "claim subject")

    note_id = new_id("n_")
    built_relationships = tuple(
        RelationshipAssertion(
            id=new_id("x_"),
            source=slugify(a["source"]),
            target=slugify(a["target"]),
            type=a["type"],
            strength=int(a.get("strength", 5)),
            description=a["description"],
        )
        for a in relationship_assertions
    )
    built_claims = tuple(
        ClaimAssertion(id=new_id("k_"), subject=slugify(c["subject"]), text=c["text"])
        for c in claim_assertions
    )
    note = Note(
        id=note_id,
        derived_from=derived_from,
        created=_timestamp(_now()),
        author="llm",
        body=content,
        entities=tuple(
            EntityInstance(
                name=slugify(e["name"]), type=e["type"], description=e["description"]
            )
            for e in entities
        ),
        relationship_assertions=built_relationships,
        claim_assertions=built_claims,
    )

    slug = slugify(content.splitlines()[0] if content.strip() else note_id)
    with session.operation({"tool": "write_note", "note": note_id}):
        path = session.store.write_note(note, slug or note_id)
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )

    return {
        "id": note_id,
        "path": _relative(session, path),
        "entities": [e.name for e in note.entities],
        "relationship_assertions": [
            {
                "id": a.id,
                "pair": f"{a.source}|{a.type}|{a.target}",
                "status": "proposed",
            }
            for a in built_relationships
        ],
        "claim_assertions": [
            {"id": c.id, "subject": c.subject, "status": "proposed"}
            for c in built_claims
        ],
        "next": "Nothing further is required. Proposals wait in review_queue.",
    }
