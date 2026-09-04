"""Tool implementations as plain functions over a Session.

Keeping these free of MCP types makes them directly testable; server.py does
nothing but wire them to the protocol.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from mindpalace.citations import extract_ids, unresolvable
from mindpalace.cluster import Community, match_lineages, partition, should_cluster
from mindpalace.config import ConfigError, add_edge_type, add_entity_type, load_config
from mindpalace.frontmatter import FrontMatterError
from mindpalace.graph.fold import MergeCycleError, resolve_merges
from mindpalace.ids import UnknownIdError, id_kind, new_id, slugify, unit_id
from mindpalace.ingest.ir import to_markdown
from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.text import parse_text
from mindpalace.ingest.units import TextUnit, split_text, to_text_units
from mindpalace.models import (
    VALIDITY_UNKNOWN,
    Capture,
    ClaimAssertion,
    CommunityReport,
    Drop,
    EntityInstance,
    Note,
    RelationshipAssertion,
    _validity,
    is_valid_validity,
    validity_precision,
)
from mindpalace.oplog import MERGE_ACTIONS, VOCABULARY_ACTIONS, VOCABULARY_KINDS
from mindpalace.index.sync import fold_notes_with_quarantine
from mindpalace.rebuild import (
    community_input_hash,
    entity_input_hash,
    mark_stale_reports,
    rebuild,
)
from mindpalace.retrieve import global_search, local_search
from mindpalace.session import Session

DEFAULT_EXTRACTION_NEXT = (
    "Read any nearest notes you need, then call write_note with this capture id. "
    "Propose a relationship only where you can give a specific rationale citing "
    "both endpoints. Proposing nothing is a valid outcome."
)
DEFAULT_INGEST_NEXT = (
    "This capture is long. Read its units by id (u_...) with `read` rather than the "
    "whole capture, then call write_note with this capture id and pass the unit ids "
    "you drew on as text_unit_ids on each entity, relationship, and claim. "
    "Proposing nothing is a valid outcome."
)


class ToolError(ValueError):
    """Raised for invalid tool input. Surfaced to the assistant verbatim."""


STRENGTH_RANGE = range(1, 11)
#: An "entity name" longer than this is a sentence the model failed to
#: reduce to a name (Utopia's `not_an_entity_name`, docs/decisions/0001 §2).
MAX_NAME_WORDS = 6


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
        "SELECT slug, type, rank, merged_from FROM entities "
        "ORDER BY rank DESC, slug LIMIT ?",
        (limit,),
    ).fetchall()
    entities = []
    for row in rows:
        try:
            page = session.store.read_entity_page(row["slug"])
        except FrontMatterError:
            # A malformed page must not break every other entity's listing
            # here (spec §10); it is already surfaced via vault_issues.
            page = None
        aliases = list(page.user.get("aliases", [])) if page else []
        # A merged-away name is an alias in every sense that matters to the
        # extractor: reuse the canonical one, never resurrect the duplicate.
        aliases += [m for m in row["merged_from"].split(",") if m]
        entities.append(
            {
                "name": row["slug"],
                "type": row["type"],
                "rank": row["rank"],
                "aliases": aliases,
            }
        )
    return entities


def _previously_dismissed(session: Session) -> list[dict]:
    reasons = session.decisions.dismissal_reasons()
    if not reasons:
        return []
    placeholders = ",".join("?" for _ in reasons)
    rows = session.conn.execute(
        f"SELECT id, source, type, proposed_type, target FROM assertions "
        f"WHERE id IN ({placeholders})",
        tuple(reasons),
    ).fetchall()
    return [
        {
            "pair": _pair(row["source"], row["type"], row["proposed_type"], row["target"]),
            "reason": reasons[row["id"]],
        }
        for row in rows
    ]


def _rebuild(session: Session, scope: str = "all"):
    """Every write tool rebuilds through here so the vocabulary and merge
    logs (docs/decisions/0001 §1, §5) reach the fold on every call."""
    return rebuild(
        session.conn,
        session.store,
        session.config,
        session.embedder,
        session.statuses(),
        scope=scope,
        **session.overlays(),
    )


class _Ledger:
    """Collects the items `write_note` refuses (docs/decisions/0001 §2).

    Dropping one item and keeping the rest is the whole point: a single
    malformed entry used to fail the entire note, and nothing recorded what
    was lost.
    """

    def __init__(self) -> None:
        self.drops: list[Drop] = []

    def drop(self, kind: str, reason: str, detail: str, example: object = None) -> None:
        self.drops.append(
            Drop(
                kind=kind,
                reason=reason,
                detail=detail,
                example=None if example is None else str(example)[:200],
            )
        )


def _text(item: dict, field: str, ledger: _Ledger, kind: str, example: object) -> str | None:
    """The field as a string, or None after recording a `bad_value` drop.

    JSON null, a list, or a number where a string belongs used to be either
    refused with a traceback or -- worse -- coerced with `str()` into a real
    entity called "none". Neither is a drop the user can read.
    """
    value = item[field]
    if isinstance(value, str):
        return value
    ledger.drop(kind, "bad_value", f"{field} must be a string, got {type(value).__name__}", example)
    return None


def _entity_type_in_graph(session: Session, slug: str) -> str | None:
    row = session.conn.execute(
        "SELECT type FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    if row is None or row["type"] == "unknown":
        return None
    return row["type"]


def _apply_signature(
    session: Session,
    source: str,
    target: str,
    edge_type: str,
    local_types: dict[str, str | None],
) -> tuple[str, str, str | None, bool]:
    """Enforce the edge type's domain/range at the write boundary
    (docs/decisions/0001 §3).

    Returns (source, target, type-or-None, direction_corrected). Endpoint
    types come from the note being written first, then the graph; an
    endpoint of unknown type never violates. Directed: if the pair is
    invalid as given but valid swapped, swap and flag. Otherwise the type is
    dropped and the endpoints and rationale stay -- the caller records the
    `domain_mismatch` drop.
    """
    spec = session.config.edge_types[edge_type]
    if spec.domain is None and spec.range is None:
        return source, target, edge_type, False

    def type_of(slug: str) -> str | None:
        if slug in local_types:
            return local_types[slug]
        return _entity_type_in_graph(session, slug)

    def fits(entity_type: str | None, allowed: tuple[str, ...] | None) -> bool:
        return allowed is None or entity_type is None or entity_type in allowed

    s_type, t_type = type_of(source), type_of(target)
    if not spec.directed:
        if fits(s_type, spec.domain) and fits(t_type, spec.domain):
            return source, target, edge_type, False
        return source, target, None, False
    if fits(s_type, spec.domain) and fits(t_type, spec.range):
        return source, target, edge_type, False
    if fits(t_type, spec.domain) and fits(s_type, spec.range):
        return target, source, edge_type, True
    return source, target, None, False


def _units_of(item: dict, ledger: _Ledger, kind: str, example: object, allowed: set[str]) -> tuple[str, ...] | None:
    """The item's `text_unit_ids`, or None after an `unknown_text_unit` drop.

    Every id must be a string naming a unit of the note's own capture
    (docs/decisions/0002 §Provenance): provenance pointing at another
    capture, or at nothing, is worse than none."""
    ids = item.get("text_unit_ids", [])
    if ids is None:
        return ()
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        ledger.drop(kind, "unknown_text_unit", "text_unit_ids must be a list of unit ids", example)
        return None
    unknown = [i for i in ids if i not in allowed]
    if unknown:
        ledger.drop(
            kind, "unknown_text_unit",
            f"{unknown} are not units of this note's capture", example,
        )
        return None
    return tuple(dict.fromkeys(ids))


def _validate_entities(
    session: Session, entities, ledger: _Ledger, allowed_units: set[str] = frozenset()
) -> list[EntityInstance]:
    built: list[EntityInstance] = []
    for entity in entities:
        example = entity.get("name") if isinstance(entity, dict) else entity
        missing = [f for f in ("name", "type", "description") if f not in entity]
        if missing:
            ledger.drop("entity", "missing_field", ", ".join(missing), example)
            continue
        name = _text(entity, "name", ledger, "entity", example)
        declared = _text(entity, "type", ledger, "entity", example)
        description = _text(entity, "description", ledger, "entity", example)
        if name is None or declared is None or description is None:
            continue
        if len(name.split()) > MAX_NAME_WORDS:
            ledger.drop(
                "entity", "not_an_entity_name",
                f"{len(name.split())} words; a name, not a sentence", name,
            )
            continue
        slug = slugify(name)
        if not slug:
            ledger.drop("entity", "empty_slug", "name normalises to nothing", name)
            continue
        units = _units_of(entity, ledger, "entity", example, allowed_units)
        if units is None:
            continue
        if declared in session.config.entity_types:
            built.append(
                EntityInstance(name=slug, type=declared, description=description, text_unit_ids=units)
            )
        else:
            built.append(
                EntityInstance(
                    name=slug, type=None, description=description,
                    proposed_type=declared, text_unit_ids=units,
                )
            )
    return built


def _validate_relationships(
    session: Session,
    assertions,
    ledger: _Ledger,
    local_types: dict[str, str | None],
    allowed_units: set[str] = frozenset(),
) -> list[RelationshipAssertion]:
    built: list[RelationshipAssertion] = []
    for assertion in assertions:
        example = (
            f"{assertion.get('source')} -> {assertion.get('target')}"
            if isinstance(assertion, dict) else assertion
        )
        missing = [f for f in ("source", "target", "type", "description") if f not in assertion]
        if missing:
            ledger.drop("relationship", "missing_field", ", ".join(missing), example)
            continue
        strength = assertion.get("strength", 5)
        if not isinstance(strength, int) or strength not in STRENGTH_RANGE:
            ledger.drop("relationship", "bad_strength", f"strength {strength!r} is not 1-10", example)
            continue
        fields = [
            _text(assertion, f, ledger, "relationship", example)
            for f in ("source", "target", "type", "description")
        ]
        if any(f is None for f in fields):
            continue
        raw_source, raw_target, raw_type, description = fields
        source, target = slugify(raw_source), slugify(raw_target)
        if not source or not target:
            ledger.drop("relationship", "empty_slug", "an endpoint normalises to nothing", example)
            continue
        if source == target:
            ledger.drop("relationship", "self_loop", "source and target are the same entity", example)
            continue
        units = _units_of(assertion, ledger, "relationship", example, allowed_units)
        if units is None:
            continue
        corrected = False
        proposed = None
        edge_type = raw_type
        if edge_type in session.config.edge_types:
            source, target, edge_type, corrected = _apply_signature(
                session, source, target, edge_type, local_types
            )
            if edge_type is None:
                proposed = raw_type
                ledger.drop(
                    "relationship", "domain_mismatch",
                    f"{proposed!r} is not declared between these entity types in "
                    f"either direction; the type was dropped, the link kept",
                    example,
                )
        else:
            proposed = raw_type
            edge_type = None
        built.append(
            RelationshipAssertion(
                id=new_id("x_"),
                source=source,
                target=target,
                type=edge_type,
                strength=int(strength),
                description=description,
                proposed_type=proposed,
                direction_corrected=corrected,
                text_unit_ids=units,
            )
        )
    return built


def _at_or_before(left: str, right: str) -> bool:
    """Compare two validity strings of possibly different precision on their
    common prefix, so "2026-03" neither precedes nor follows "2026-03-15"."""
    n = min(len(left), len(right))
    return left[:n] <= right[:n]


def _validate_claims(
    session: Session, claims, ledger: _Ledger, allowed_units: set[str] = frozenset()
) -> list[ClaimAssertion]:
    built: list[ClaimAssertion] = []
    for claim in claims:
        example = claim.get("text") if isinstance(claim, dict) else claim
        missing = [f for f in ("subject", "text") if f not in claim]
        if missing:
            ledger.drop("claim", "missing_field", ", ".join(missing), example)
            continue
        raw_subject = _text(claim, "subject", ledger, "claim", example)
        text = _text(claim, "text", ledger, "claim", example)
        if raw_subject is None or text is None:
            continue
        if not text.strip():
            ledger.drop("claim", "empty_text", "claim text is empty", raw_subject)
            continue
        subject = slugify(raw_subject)
        if not subject:
            ledger.drop("claim", "empty_slug", "subject normalises to nothing", example)
            continue
        valid_from = _validity_string(claim.get("valid_from"))
        valid_to = _validity_string(claim.get("valid_to"))
        if not is_valid_validity(valid_from, allow_unknown=False):
            ledger.drop(
                "claim", "bad_validity",
                f"valid_from {valid_from!r} is not YYYY, YYYY-MM or YYYY-MM-DD", example,
            )
            continue
        if not is_valid_validity(valid_to, allow_unknown=True):
            ledger.drop(
                "claim", "bad_validity",
                f"valid_to {valid_to!r} is not YYYY, YYYY-MM, YYYY-MM-DD or 'unknown'",
                example,
            )
            continue
        if (
            valid_from is not None
            and valid_to not in (None, VALIDITY_UNKNOWN)
            and not _at_or_before(valid_from, valid_to)
        ):
            ledger.drop(
                "claim", "bad_validity",
                f"valid_from {valid_from} is after valid_to {valid_to}", example,
            )
            continue
        units = _units_of(claim, ledger, "claim", example, allowed_units)
        if units is None:
            continue
        supersedes = claim.get("supersedes")
        if supersedes is not None and not isinstance(supersedes, str):
            ledger.drop("claim", "bad_value", "supersedes must be a claim id string", example)
            continue
        if supersedes is not None:
            known = session.conn.execute(
                "SELECT 1 FROM claims WHERE id = ?", (supersedes,)
            ).fetchone()
            if known is None:
                ledger.drop(
                    "claim", "unknown_supersedes",
                    f"{supersedes!r} is not an existing claim id", example,
                )
                continue
        built.append(
            ClaimAssertion(
                id=new_id("k_"),
                subject=subject,
                text=text,
                valid_from=valid_from,
                valid_to=valid_to,
                supersedes=supersedes,
                text_unit_ids=units,
            )
        )
    return built


def _validity_string(value: object) -> str | None:
    """A date-like value as the string the precision rules read. Anything
    that is not a date, datetime, int year, or string is left as-is for
    `is_valid_validity` to reject."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (datetime, date)) or isinstance(value, int) and not isinstance(value, bool):
        return _validity(value)
    return value


def _pair(source: str, edge_type: str | None, proposed: str | None, target: str) -> str:
    label = edge_type if edge_type is not None else f"(untyped: {proposed})"
    return f"{source}|{label}|{target}"


def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"1 {noun}"
    return f"{count} {'entities' if noun == 'entity' else noun + 's'}"


def _landed(
    entities: list[EntityInstance],
    relationships: list[RelationshipAssertion],
    claims: list[ClaimAssertion],
    drops: list[Drop],
) -> str:
    """One line saying what actually entered the vault
    (docs/decisions/0001 §7). The assistant shows this, not its intent."""
    untyped_entities = sum(1 for e in entities if e.type is None)
    untyped_rels = sum(1 for r in relationships if r.type is None)
    corrected = sum(1 for r in relationships if r.direction_corrected)
    parts = []
    entity_notes = [f"{untyped_entities} untyped"] if untyped_entities else []
    parts.append(_plural(len(entities), "entity") + (f" ({', '.join(entity_notes)})" if entity_notes else ""))
    rel_notes = []
    if untyped_rels:
        rel_notes.append(f"{untyped_rels} untyped")
    if corrected:
        rel_notes.append(f"{corrected} direction-corrected")
    parts.append(_plural(len(relationships), "relationship") + (f" ({', '.join(rel_notes)})" if rel_notes else ""))
    parts.append(_plural(len(claims), "claim"))
    return f"{', '.join(parts)}; {len(drops)} dropped"


def _relationship_payload(a: RelationshipAssertion) -> dict:
    return {
        "id": a.id,
        "pair": _pair(a.source, a.type, a.proposed_type, a.target),
        "source": a.source,
        "target": a.target,
        "type": a.type,
        "proposed_type": a.proposed_type,
        "direction_corrected": a.direction_corrected,
        "status": "proposed",
    }


def _claim_payload(c: ClaimAssertion) -> dict:
    return {
        "id": c.id,
        "subject": c.subject,
        "valid_from": c.valid_from,
        "valid_to": c.valid_to,
        "supersedes": c.supersedes,
        "status": "proposed",
    }


def _spans_as_units(spans: tuple[tuple[int, int], ...], text: str) -> list[TextUnit]:
    if not spans:
        return [TextUnit(0, 0, len(text), None)]
    return [TextUnit(i, start, end, None) for i, (start, end) in enumerate(spans)]


def _extraction_payload(session: Session, capture: Capture, units: list[TextUnit]) -> dict:
    """What every capture entry point hands back so the assistant can extract:
    the nearest existing material, the vocabularies, and the instruction.
    A multi-unit capture also gets a unit preview and the long-form
    instruction (docs/decisions/0002 §Tool surface)."""
    nearest = local_search(
        session.conn, session.embedder, capture.text[:2000], session.config, k=5
    )
    preview = []
    for unit in units[:5]:
        text = capture.text[unit.start:unit.end].strip()
        preview.append(
            {
                "id": unit_id(capture.id, unit.ordinal),
                "locator": (
                    f"{unit.locator.kind} {unit.locator.label}" if unit.locator else None
                ),
                "first_line": text.splitlines()[0][:120] if text else "",
            }
        )
    long = len(units) > 1
    templates = session.config.templates
    return {
        "units": max(len(units), 1),
        "unit_preview": preview,
        "nearest": nearest["hits"],
        "known_entities": _known_entities(session),
        "previously_dismissed": _previously_dismissed(session),
        "entity_types": session.config.entity_types,
        "edge_vocabulary": _edge_vocabulary(session),
        "next": (
            templates.get("ingest_next", DEFAULT_INGEST_NEXT)
            if long
            else templates.get("extraction_next", DEFAULT_EXTRACTION_NEXT)
        ),
    }


def save_capture(
    session: Session, text: str, why: str | None = None, source: str = "manual"
) -> dict:
    if not text.strip():
        raise ToolError("capture text is empty")

    when = _now()
    capture_id = new_id("c_")
    # The typed text is stored verbatim, never re-rendered: the user's own
    # words are not rewritten. The IR is used only to decide the unit spans,
    # which are then mapped back onto the original (docs/decisions/0002 §IR).
    document = parse_text("capture.txt", text.encode("utf-8"))
    spans = tuple(split_text(text)) if len(to_text_units(document)) > 1 else ()
    capture = Capture(
        id=capture_id,
        created=_timestamp(when),
        source=source,
        why=why,
        text=text,
        sha256=document.source.sha256,
        mime="text/plain",
        parser="text",
        parser_version=document.source.parser_version,
        units=spans,
    )

    with session.operation({"tool": "save_capture", "capture": capture_id}):
        path = session.store.write_capture(capture, when)
        session.resync()

    payload = {"id": capture_id, "path": _relative(session, path)}
    payload.update(_extraction_payload(session, capture, _spans_as_units(spans, text)))
    return payload


def _capture_from_document(document, capture_id: str, when: datetime, source: str, why: str | None):
    body = to_markdown(document)
    units = to_text_units(document)
    spans = tuple((u.start, u.end) for u in units) if len(units) > 1 else ()
    capture = Capture(
        id=capture_id,
        created=_timestamp(when),
        source=source,
        why=why,
        text=body.rstrip("\n"),
        title=document.title,
        sha256=document.source.sha256,
        mime=document.source.mime,
        parser=document.source.parser,
        parser_version=document.source.parser_version,
        metadata=dict(document.metadata),
        units=spans,
    )
    return capture, units


def ingest_file(
    session: Session, path: str, why: str | None = None, source: str = "file"
) -> dict:
    """Ingest a local file through the Document IR (docs/decisions/0002).

    The original bytes are kept as a content-addressed attachment, the
    rendered markdown becomes the capture, and the unit offsets go into the
    capture's front-matter. A file already in the vault (same sha256) returns
    its existing capture and writes nothing.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise ToolError(f"no such file: {path}")
    data = file.read_bytes()
    _require(bool(data), f"{file.name} is empty")
    document = parse_bytes(file.name, data)

    existing = session.store.find_capture_by_sha256(document.source.sha256)
    if existing is not None:
        payload = {
            "id": existing.id,
            "path": None,
            "title": existing.title,
            "attachment": existing.attachment,
            "parser": existing.parser,
            "parser_error": existing.metadata.get("parser_error"),
            "duplicate": True,
        }
        payload.update(
            _extraction_payload(session, existing, _spans_as_units(existing.units, existing.text))
        )
        return payload

    when = _now()
    capture_id = new_id("c_")
    capture, units = _capture_from_document(document, capture_id, when, source, why)
    ext = file.suffix.lstrip(".").lower()
    with session.operation({"tool": "ingest_file", "capture": capture_id}):
        attachment = session.store.write_attachment(document.source.sha256, ext, data)
        capture = replace(capture, attachment=_relative(session, attachment))
        written = session.store.write_capture(capture, when)
        session.resync()

    payload = {
        "id": capture_id,
        "path": _relative(session, written),
        "title": capture.title,
        "attachment": capture.attachment,
        "parser": capture.parser,
        "parser_error": capture.metadata.get("parser_error"),
        "duplicate": False,
    }
    payload.update(_extraction_payload(session, capture, units))
    return payload


def write_note(
    session: Session,
    derived_from: str,
    content: str,
    entities: list[dict] | tuple = (),
    relationship_assertions: list[dict] | tuple = (),
    claim_assertions: list[dict] | tuple = (),
) -> dict:
    """Note-level failures raise; item-level failures are dropped and
    recorded in the note's `drops:` ledger (docs/decisions/0001 §2)."""
    _require(bool(content.strip()), "note content is empty")

    known_captures = {capture.id for capture in session.store.iter_captures()}
    _require(derived_from in known_captures, f"no such capture: {derived_from}")

    ledger = _Ledger()
    allowed_units = {
        row["id"]
        for row in session.conn.execute(
            "SELECT id FROM text_units WHERE capture_id = ?", (derived_from,)
        )
    }
    built_entities = _validate_entities(session, entities, ledger, allowed_units)
    local_types = {e.name: e.type for e in built_entities}
    built_relationships = _validate_relationships(
        session, relationship_assertions, ledger, local_types, allowed_units
    )
    built_claims = _validate_claims(session, claim_assertions, ledger, allowed_units)

    note_id = new_id("n_")
    note = Note(
        id=note_id,
        derived_from=derived_from,
        created=_timestamp(_now()),
        author="llm",
        body=content,
        entities=tuple(built_entities),
        relationship_assertions=tuple(built_relationships),
        claim_assertions=tuple(built_claims),
        drops=tuple(ledger.drops),
    )

    slug = slugify(content.splitlines()[0] if content.strip() else note_id)
    with session.operation({"tool": "write_note", "note": note_id}):
        path = session.store.write_note(note, slug or note_id)
        _rebuild(session)

    landed = _landed(built_entities, built_relationships, built_claims, ledger.drops)
    return {
        "id": note_id,
        "path": _relative(session, path),
        "entities": [
            {"name": e.name, "type": e.type, "proposed_type": e.proposed_type}
            for e in built_entities
        ],
        "relationship_assertions": [_relationship_payload(a) for a in built_relationships],
        "claim_assertions": [_claim_payload(c) for c in built_claims],
        "dropped": [
            {"kind": d.kind, "reason": d.reason, "detail": d.detail, "example": d.example}
            for d in ledger.drops
        ],
        "landed": landed,
        "next": (
            f"Tell the user what landed, not what you intended: {landed}. "
            f"An untyped item kept the wording you gave it and waits as a "
            f"vocabulary proposal; a direction-corrected one was swapped to fit "
            f"the edge type's signature. Proposals wait in review_queue."
        ),
    }


def _tables(session: Session):
    """Fold the current source set. Several tools need it for evidence hashing.

    Quarantine-aware: a note `fold` rejects (unknown edge type, duplicate
    assertion id) must not take `cluster`, `write_entity_description`, or
    `write_community_report` down with it (spec §10). See
    `mindpalace.index.sync.fold_notes_with_quarantine`.
    """
    overlays = session.overlays()
    notes, notes_by_id, tables, _issues = fold_notes_with_quarantine(
        session.store,
        session.config,
        session.statuses(),
        overlays["adoptions"],
        overlays["merges"],
    )
    return notes, notes_by_id, tables


def read(session: Session, identifier: str) -> dict:
    try:
        kind = id_kind(identifier)
    except UnknownIdError as exc:
        raise ToolError(str(exc)) from exc

    if kind == "capture":
        for capture in session.store.iter_captures():
            if capture.id == identifier:
                return {"id": identifier, "kind": "capture", "text": capture.text}
    elif kind == "note":
        for note in session.store.iter_notes():
            if note.id == identifier:
                return {"id": identifier, "kind": "note", "text": note.body}
    elif kind == "entity":
        try:
            page = session.store.read_entity_page(
                _canonical_slug(session, identifier.removeprefix("e_"))
            )
        except FrontMatterError as exc:
            # A malformed page must surface as a ToolError the assistant can
            # act on, not an uncaught traceback out of the tool boundary
            # (spec §10); it is also already recorded as a vault_issue.
            raise ToolError(f"{identifier}: entity page is malformed: {exc}") from exc
        if page is not None:
            return {
                "id": identifier,
                "kind": "entity",
                "text": page.description,
                "stale": page.stale,
                "related": page.related,
                "user": page.user,
            }
    elif kind == "text_unit":
        row = session.conn.execute(
            "SELECT capture_id, ordinal, locator, text FROM text_units WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is not None:
            return {
                "id": identifier,
                "kind": "text_unit",
                "capture": row["capture_id"],
                "ordinal": row["ordinal"],
                "locator": row["locator"],
                "text": row["text"],
            }
    elif kind == "community":
        try:
            report = session.store.read_report(identifier)
        except FrontMatterError as exc:
            raise ToolError(
                f"{identifier}: community report is malformed: {exc}"
            ) from exc
        if report is not None:
            return {
                "id": identifier,
                "kind": "community",
                "title": report.title,
                "text": report.summary,
                "findings": report.findings,
                "stale": report.stale,
            }
    raise ToolError(f"{identifier} not found")


def neighbors(
    session: Session,
    identifier: str,
    depth: int = 1,
    edge_types: list[str] | None = None,
) -> dict:
    frontier = {identifier.removeprefix("e_")}
    seen = set(frontier)
    collected: list[dict] = []

    for _ in range(max(1, depth)):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        rows = session.conn.execute(
            f"SELECT source, target, type, weight FROM aggregates "
            f"WHERE traversable = 1 AND (source IN ({placeholders}) "
            f"OR target IN ({placeholders}))",
            (*frontier, *frontier),
        ).fetchall()

        next_frontier: set[str] = set()
        for row in rows:
            if edge_types and row["type"] not in edge_types:
                continue
            for slug in (row["source"], row["target"]):
                if slug in seen:
                    continue
                seen.add(slug)
                next_frontier.add(slug)
                collected.append(
                    {"slug": slug, "type": row["type"], "weight": row["weight"]}
                )
        frontier = next_frontier

    grouped: dict[str, list[dict]] = {}
    for neighbour in collected:
        grouped.setdefault(neighbour["type"], []).append(neighbour)

    return {"id": identifier, "neighbours": collected, "by_type": grouped}


def _canonical_slug(session: Session, slug: str) -> str:
    """Follow `merges.jsonl` to the slug the graph holds
    (docs/decisions/0001 §5). A cycle in a hand-edited log is already a
    vault_issue, so here it just means "no merge applies"."""
    try:
        return resolve_merges(session.merges.merges()).get(slug, slug)
    except MergeCycleError:
        return slug


def _resolve_slug(session: Session, name: str) -> str | None:
    """Exact slug always wins outright; only alias-to-alias ambiguity raises.

    Two entities claiming the same alias is a vault_issue (spec §8.5), never
    silently resolved in favour of one -- aliases explicitly do not merge
    identities. Picking a winner here would hand the caller the wrong
    entity's data with no signal anything was wrong; an error the assistant
    can relay, naming every candidate, is the only actionable outcome.
    `mindpalace.index.sync._ambiguous_alias_issues` surfaces the same
    collision proactively through `review_queue`, before anyone happens to
    look the ambiguous name up.
    """
    slug = _canonical_slug(session, slugify(name))
    row = session.conn.execute(
        "SELECT slug FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    if row is not None:
        return row["slug"]
    matches: list[str] = []
    for page in session.store.iter_entity_pages():
        # `iter_entity_pages` already skips a page whose `user:` isn't a
        # mapping (`read_entity_page` raises `FrontMatterError` for that,
        # caught there) -- this is defence in depth against the same
        # unguarded `.get` that bricked `sync` (spec §10 regression), not a
        # path that should be reachable in practice.
        user = page.user if isinstance(page.user, dict) else {}
        aliases = {slugify(alias) for alias in user.get("aliases", [])}
        if slug in aliases:
            matches.append(page.slug)
    if len(matches) > 1:
        candidates = sorted(set(matches))
        raise ToolError(
            f"alias {name!r} is ambiguous: claimed by more than one entity "
            f"{candidates}; aliases do not merge identities"
        )
    return matches[0] if matches else None


def _claim_row(record) -> dict:
    return {
        "id": record["id"],
        "text": record["text"],
        "status": record["status"],
        "valid_from": record["valid_from"],
        "valid_to": record["valid_to"],
        "valid_from_precision": validity_precision(record["valid_from"]),
        "valid_to_precision": validity_precision(record["valid_to"]),
        "supersedes": record["supersedes"],
    }


def _holds_at(claim: dict, as_of: str) -> bool:
    """Whether a claim's validity interval contains `as_of`
    (docs/decisions/0001 §4). No bound means unbounded on that side;
    `valid_to: unknown` means it ended some time we cannot name, so it is
    never excluded on that evidence."""
    if claim["valid_from"] is not None and not _at_or_before(claim["valid_from"], as_of):
        return False
    valid_to = claim["valid_to"]
    if valid_to is None or valid_to == VALIDITY_UNKNOWN:
        return True
    return _at_or_before(as_of, valid_to)


def get_entity(session: Session, name: str, as_of: str | None = None) -> dict:
    slug = _resolve_slug(session, name)
    if slug is None:
        raise ToolError(f"no entity matching {name!r}")
    if as_of is not None:
        _require(
            is_valid_validity(as_of, allow_unknown=False),
            f"as_of must be YYYY, YYYY-MM or YYYY-MM-DD, got {as_of!r}",
        )

    try:
        page = session.store.read_entity_page(slug)
    except FrontMatterError:
        # A malformed page must not stop `get_entity` from returning what
        # the graph still knows (type, rank, neighbours) -- it degrades to
        # the same shape as "no page written yet" (spec §10).
        page = None
    row = session.conn.execute(
        "SELECT type, rank, merged_from FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    claims = [
        _claim_row(record)
        for record in session.conn.execute(
            "SELECT id, text, status, valid_from, valid_to, supersedes "
            "FROM claims WHERE subject = ? ORDER BY id",
            (slug,),
        )
    ]
    if as_of is not None:
        claims = [c for c in claims if _holds_at(c, as_of)]
    return {
        "slug": slug,
        "type": row["type"] if row else (page.type if page else "unknown"),
        "rank": row["rank"] if row else 0,
        "merged_from": [m for m in row["merged_from"].split(",") if m] if row else [],
        "description": page.description if page else "",
        "stale": page.stale if page else True,
        "user": page.user if page else {},
        "as_of": as_of,
        "claims": claims,
        "neighbours": neighbors(session, f"e_{slug}")["neighbours"],
    }


def search_local(
    session: Session, query: str, k: int = 8, expand_graph: bool = False
) -> dict:
    return local_search(
        session.conn, session.embedder, query, session.config, k, expand_graph
    )


def search_global(session: Session, query: str) -> dict:
    return global_search(
        session.conn, session.store, session.embedder, query, session.config
    )


def graph_stats(session: Session) -> dict:
    def count(sql: str) -> int:
        return session.conn.execute(sql).fetchone()[0]

    entities = count("SELECT COUNT(*) FROM entities")
    threshold = session.config.thresholds.cluster_activation_entities
    communities = count("SELECT COUNT(*) FROM communities")
    stale_pages = sum(1 for page in session.store.iter_entity_pages() if page.stale)
    stale_reports = sum(1 for report in session.store.iter_reports() if report.stale)

    return {
        "entities": entities,
        "orphans": count("SELECT COUNT(*) FROM entities WHERE rank = 0"),
        "assertions": count("SELECT COUNT(*) FROM assertions"),
        "aggregates": count("SELECT COUNT(*) FROM aggregates"),
        "traversable_aggregates": count(
            "SELECT COUNT(*) FROM aggregates WHERE traversable = 1"
        ),
        "claims": count("SELECT COUNT(*) FROM claims"),
        "text_units": count("SELECT COUNT(*) FROM text_units"),
        "attachments": sum(1 for p in session.paths.attachments.glob("*") if p.is_file()),
        "superseded_claims": count("SELECT COUNT(*) FROM claims WHERE status = 'superseded'"),
        "untyped_assertions": count("SELECT COUNT(*) FROM assertions WHERE type IS NULL"),
        "vocabulary_proposals": count("SELECT COUNT(*) FROM vocabulary_proposals"),
        "drops": {
            "total": count("SELECT COUNT(*) FROM drops"),
            "by_reason": {
                row["reason"]: row["n"]
                for row in session.conn.execute(
                    "SELECT reason, COUNT(*) AS n FROM drops GROUP BY reason ORDER BY reason"
                )
            },
        },
        "merges": len(session.merges.merges()),
        "communities": communities,
        "stale_entity_pages": stale_pages,
        "stale_reports": stale_reports,
        "vault_issues": count("SELECT COUNT(*) FROM vault_issues"),
        "clustering": {
            "active": communities > 0,
            "eligible": should_cluster(entities, threshold),
            "entity_count": entities,
            "threshold": threshold,
            "remaining": max(0, threshold - entities),
        },
        "embedder": {
            "model_id": session.embedder.model_id,
            "kind": session.config.embedder.get("kind"),
            "sends_data_off_machine": session.config.embedder.get("kind") == "cloud",
        },
    }


VALID_ACTIONS = {"confirm": "confirmed", "dismiss": "dismissed"}


def propose_relationship(
    session: Session,
    source: str,
    target: str,
    type: str,
    description: str,
    strength: int = 5,
) -> dict:
    """One assertion, so a hard failure raises rather than dropping: there
    would be nothing left to write. An unknown type is not a failure -- it
    is kept as a proposal (docs/decisions/0001 §1) -- and a signature
    violation is corrected or untyped exactly as in `write_note` (§3)."""
    _require(
        isinstance(strength, int) and strength in STRENGTH_RANGE,
        f"strength must be an integer 1-10, got {strength!r}",
    )
    _require(bool(description.strip()), "a proposed relationship needs a rationale")
    source_slug = _require_slug(source, "relationship source")
    target_slug = _require_slug(target, "relationship target")
    _require(source_slug != target_slug, "source and target are the same entity")

    ledger = _Ledger()
    [assertion] = _validate_relationships(
        session,
        [
            {
                "source": source_slug,
                "target": target_slug,
                "type": type,
                "description": description,
                "strength": strength,
            }
        ],
        ledger,
        {},
    )
    note = Note(
        id=new_id("n_"),
        derived_from=None,
        created=_timestamp(_now()),
        author="user",
        body=f"Relationship proposed outside extraction: {description}",
        relationship_assertions=(assertion,),
        drops=tuple(ledger.drops),
    )
    with session.operation({"tool": "propose_relationship", "note": note.id}):
        session.store.write_note(note, f"link-{assertion.source}-{assertion.target}")
        _rebuild(session)
    payload = _relationship_payload(assertion)
    payload["note"] = note.id
    payload["dropped"] = [
        {"kind": d.kind, "reason": d.reason, "detail": d.detail, "example": d.example}
        for d in ledger.drops
    ]
    payload["landed"] = _landed([], [assertion], [], ledger.drops)
    return payload


def adopt_type(
    session: Session,
    kind: str,
    proposed: str,
    name: str,
    directed: bool | None = None,
    cluster_weight: float = 1.0,
    domain: list[str] | None = None,
    range: list[str] | None = None,
    action: str = "adopt",
) -> dict:
    """Promote a proposed wording into the vocabulary (docs/decisions/0001 §1).

    Adds `name` to MINDPALACE.md if it is not there yet, then records
    `proposed -> name` in `vocabulary.jsonl`; the next fold retypes every
    assertion (or entity) that carried that wording. Note files are never
    rewritten. `action="revoke"` withdraws the mapping and leaves the type
    in the config: a type that has been used is history, not clutter.
    """
    _require(kind in VOCABULARY_KINDS, f"kind must be one of {sorted(VOCABULARY_KINDS)}, got {kind!r}")
    _require(action in VOCABULARY_ACTIONS, f"action must be one of {sorted(VOCABULARY_ACTIONS)}, got {action!r}")
    key = _require_slug(proposed, "proposed wording")
    _require(bool(name.strip()), "the adopted type needs a name")

    waiting = session.conn.execute(
        "SELECT ids FROM vocabulary_proposals WHERE kind = ? AND proposed = ?",
        (kind, key),
    ).fetchone()
    waiting_ids = [i for i in waiting["ids"].split(",") if i] if waiting else []

    # Everything that can be refused is refused before the operation frame
    # opens: a `begin` with no `commit` replays as a crash on the next open.
    new_edge = action == "adopt" and kind == "edge" and name not in session.config.edge_types
    new_entity = action == "adopt" and kind == "entity" and name not in session.config.entity_types
    if new_edge:
        _require(directed is not None, f"{name!r} is a new edge type; say whether it is directed")
        for end, allowed in (("domain", domain), ("range", range)):
            unknown = [t for t in (allowed or []) if t not in session.config.entity_types]
            _require(not unknown, f"{end} names unknown entity type(s) {unknown}")
        _require(directed or not range, "a symmetric edge type takes domain only, not range")

    with session.operation({"tool": "adopt_type", "kind": kind, "proposed": key, "action": action}) as op_id:
        if new_edge or new_entity:
            try:
                if new_edge:
                    add_edge_type(
                        session.paths.mindpalace_md, name,
                        directed=directed, cluster_weight=cluster_weight,
                        domain=domain, range=range,
                    )
                else:
                    add_entity_type(session.paths.mindpalace_md, name)
            except ConfigError as exc:
                raise ToolError(str(exc)) from exc
            session.config = load_config(session.paths.mindpalace_md)
        session.vocabulary.append(kind, key, name, action, "adopt_type", op_id)
        _rebuild(session)

    return {
        "kind": kind,
        "proposed": key,
        "adopted": name,
        "action": action,
        "retyped": waiting_ids,
    }


def merge_entities(
    session: Session,
    duplicate: str,
    canonical: str,
    action: str = "merge",
    reason: str | None = None,
) -> dict:
    """Record an identity decision in `merges.jsonl` (docs/decisions/0001 §5).

    `merge` folds `duplicate` into `canonical` on every rebuild from now on;
    `unmerge` reverses it; `keep` says the two are different so the
    similarity lint stops asking. The note files are untouched, so the
    decision is exactly as reversible as any other line in a log.
    """
    _require(action in MERGE_ACTIONS, f"action must be one of {sorted(MERGE_ACTIONS)}, got {action!r}")
    dup = _require_slug(duplicate, "duplicate")
    canon = _require_slug(canonical, "canonical")
    _require(dup != canon, f"{dup!r} cannot be merged into itself")

    if action == "merge":
        current = session.merges.merges()
        known = {row["slug"] for row in session.conn.execute("SELECT slug FROM entities")}
        known |= set(current) | set(current.values())
        for slug in (dup, canon):
            _require(slug in known, f"no entity {slug!r} in the graph")
        try:
            resolve_merges({**current, dup: canon})
        except MergeCycleError as exc:
            raise ToolError(
                f"merging {dup!r} into {canon!r} would create a cycle: {exc}"
            ) from exc

    with session.operation({"tool": "merge_entities", "duplicate": dup, "canonical": canon, "action": action}) as op_id:
        session.merges.append(dup, canon, action, "merge_entities", op_id, reason)
        _rebuild(session)

    status = {"merge": "merged", "unmerge": "unmerged", "keep": "kept"}[action]
    return {"duplicate": dup, "canonical": canon, "status": status, "reason": reason}


def resolve_assertion(
    session: Session, identifier: str, action: str, reason: str | None = None
) -> dict:
    if action not in VALID_ACTIONS:
        raise ToolError(f"action must be 'confirm' or 'dismiss', got {action!r}")

    exists = session.conn.execute(
        "SELECT 1 FROM assertions WHERE id = ? UNION SELECT 1 FROM claims WHERE id = ?",
        (identifier, identifier),
    ).fetchone()
    if exists is None:
        raise ToolError(f"no such assertion: {identifier}")

    with session.operation(
        {"tool": "resolve_assertion", "assertion": identifier, "action": action}
    ) as op_id:
        session.decisions.append(identifier, action, "resolve_assertion", op_id, reason)
        _rebuild(session)

    return {"id": identifier, "status": VALID_ACTIONS[action], "reason": reason}


def _endpoint_snippet(session: Session, slug: str) -> str:
    """One line describing an entity, so a reviewer needs no extra `read` call."""
    try:
        page = session.store.read_entity_page(slug)
    except FrontMatterError:
        page = None
    if page is not None and page.description.strip():
        return page.description.strip().splitlines()[0][:200]
    row = session.conn.execute(
        "SELECT substr(text, 1, 200) AS snippet FROM docs WHERE doc_id = ?",
        (f"e_{slug}",),
    ).fetchone()
    if row is not None and row["snippet"]:
        return row["snippet"]
    return "(no description written yet)"


def review_queue(session: Session, limit: int = 20) -> dict:
    """Proposals are ranked by asserted strength.

    Not by "retrieval score": there is no query here to score anything against,
    so such a number would have nothing behind it.
    """
    proposals = [
        {
            "id": row["id"],
            "kind": "relationship",
            "pair": _pair(row["source"], row["type"], row["proposed_type"], row["target"]),
            "type": row["type"],
            "proposed_type": row["proposed_type"],
            "direction_corrected": bool(row["direction_corrected"]),
            "strength": row["strength"],
            "why": row["description"],
            "note": row["note_id"],
            "source_snippet": _endpoint_snippet(session, row["source"]),
            "target_snippet": _endpoint_snippet(session, row["target"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, source, target, type, proposed_type, "
            "direction_corrected, strength, description "
            "FROM assertions WHERE status = 'proposed' "
            "ORDER BY strength DESC, id LIMIT ?",
            (limit,),
        )
    ]
    proposals += [
        {
            "id": row["id"],
            "kind": "claim",
            "subject": row["subject"],
            "text": row["text"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "supersedes": row["supersedes"],
            "note": row["note_id"],
            "subject_snippet": _endpoint_snippet(session, row["subject"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, subject, text, valid_from, valid_to, supersedes "
            "FROM claims WHERE status = 'proposed' ORDER BY id LIMIT ?",
            (limit,),
        )
    ]
    # Wordings the vocabulary has no type for (docs/decisions/0001 §1). The
    # remedy is `adopt_type`; the count says which ones are worth it.
    vocabulary = [
        {
            "kind": row["kind"],
            "proposed": row["proposed"],
            "count": row["count"],
            "example": row["example"],
            "ids": [i for i in row["ids"].split(",") if i],
        }
        for row in session.conn.execute(
            "SELECT kind, proposed, count, example, ids FROM vocabulary_proposals "
            "ORDER BY count DESC, kind, proposed LIMIT ?",
            (limit,),
        )
    ]
    drops = [
        {
            "note": row["note_id"],
            "kind": row["kind"],
            "reason": row["reason"],
            "detail": row["detail"],
            "example": row["example"],
        }
        for row in session.conn.execute(
            "SELECT note_id, kind, reason, detail, example FROM drops "
            "ORDER BY note_id DESC LIMIT ?",
            (limit,),
        )
    ]

    issues = [
        {"path": row["path"], "kind": row["kind"], "detail": row["detail"]}
        for row in session.conn.execute(
            "SELECT path, kind, detail FROM vault_issues LIMIT ?", (limit,)
        )
    ]
    issues += [
        {"path": f"entities/{page.slug}.md", "kind": "stale_prose", "detail": "needs rewrite"}
        for page in session.store.iter_entity_pages()
        if page.stale
    ]
    issues += [
        {
            "path": f"communities/{report.lineage_id}",
            "kind": "stale_report",
            "detail": "membership changed since this was written",
        }
        for report in session.store.iter_reports()
        if report.stale
    ]

    return {
        "proposals": proposals,
        "vocabulary": vocabulary,
        "drops": drops,
        "vault_issues": issues,
    }


def _stored_communities(session: Session) -> list[Community]:
    return [
        Community(
            lineage_id=row["lineage_id"],
            level=row["level"],
            members=frozenset(row["members"].split(",")) if row["members"] else frozenset(),
            parent=row["parent"],
        )
        for row in session.conn.execute(
            "SELECT lineage_id, level, parent, members FROM communities"
        )
    ]


def cluster_tool(session: Session, force: bool = False) -> dict:
    stats = graph_stats(session)
    threshold = session.config.thresholds.cluster_activation_entities
    if not force and not should_cluster(stats["entities"], threshold):
        return {
            "clustered": False,
            "communities": [],
            "note": (
                f"{stats['entities']} entities; clustering activates at the "
                f"threshold of {threshold}. Pass force=true to run anyway."
            ),
        }

    notes, notes_by_id, tables = _tables(session)
    fresh = partition(tables.aggregates, session.config)
    matched = match_lineages(
        fresh,
        _stored_communities(session),
        session.config.thresholds.community_lineage_jaccard,
    )

    with session.operation({"tool": "cluster"}):
        with session.conn:
            session.conn.execute("DELETE FROM communities")
            session.conn.executemany(
                "INSERT INTO communities (lineage_id, level, parent, members) "
                "VALUES (?, ?, ?, ?)",
                [
                    (c.lineage_id, c.level, c.parent, ",".join(sorted(c.members)))
                    for c in matched
                ],
            )
        # Reclustering must flip affected reports to stale immediately. Deferring
        # it to the next rebuild would let global_search serve an obsolete report
        # as `present` in the meantime.
        stale_reports = mark_stale_reports(session.conn, session.store, tables)

    payload = []
    for community in matched:
        try:
            report = session.store.read_report(community.lineage_id)
        except FrontMatterError:
            # A malformed report is treated the same as no report: it needs
            # rewriting either way, and `write_community_report` always
            # replaces the file wholesale, so this cannot destroy anything
            # a fresh write wasn't already about to replace.
            report = None
        expected = community_input_hash(sorted(community.members), tables, notes_by_id)
        payload.append(
            {
                "lineage_id": community.lineage_id,
                "level": community.level,
                "members": sorted(community.members),
                # `report.stale` too, not just a hash mismatch: `rebuild`
                # sets both `stale=True` AND `input_hash=expected` together
                # when evidence moves (mark_stale_reports), so a hash-only
                # comparison here reads `stale: True` but `needs_report:
                # False` for a report that flipped stale and never
                # recovers (Important 4).
                "needs_report": (
                    report is None or report.stale or report.input_hash != expected
                ),
                "aggregates": [
                    {
                        "pair": f"{a.source}|{a.type}|{a.target}",
                        "weight": a.weight,
                    }
                    for a in tables.aggregates.values()
                    if a.traversable
                    and a.source in community.members
                    and a.target in community.members
                ],
            }
        )

    return {
        "clustered": True,
        "communities": payload,
        "reports_marked_stale": stale_reports,
        "instructions": session.config.templates.get("report_next", ""),
    }


def write_community_report(
    session: Session,
    lineage_id: str,
    title: str,
    summary: str,
    rank: float,
    findings: list[dict],
    cites: list[str],
) -> dict:
    row = session.conn.execute(
        "SELECT members FROM communities WHERE lineage_id = ?", (lineage_id,)
    ).fetchone()
    if row is None:
        raise ToolError(f"no such community: {lineage_id}")

    def resolves(identifier: str) -> bool:
        """Every cited id must point at something that exists — including notes
        and captures, which the first version waved through unchecked."""
        try:
            kind = id_kind(identifier)
        except UnknownIdError:
            return False
        if kind == "entity":
            return (
                session.conn.execute(
                    "SELECT 1 FROM entities WHERE slug = ?",
                    (identifier.removeprefix("e_"),),
                ).fetchone()
                is not None
            )
        if kind == "relationship_assertion":
            table = "assertions"
        elif kind == "claim_assertion":
            table = "claims"
        else:  # note, capture, community — all indexed as documents
            return (
                session.conn.execute(
                    "SELECT 1 FROM docs WHERE doc_id = ?", (identifier,)
                ).fetchone()
                is not None
            )
        return (
            session.conn.execute(
                f"SELECT 1 FROM {table} WHERE id = ?", (identifier,)
            ).fetchone()
            is not None
        )

    cited = list(cites) + extract_ids(summary)
    for index, finding in enumerate(findings):
        grounded = extract_ids(finding.get("summary", "")) + extract_ids(
            finding.get("explanation", "")
        )
        _require(
            bool(grounded),
            f"finding {index} carries no [Data: …] citation; every finding must "
            f"name the evidence it rests on",
        )
        cited.extend(grounded)

    _require(bool(cited), "a report must cite the evidence it was written from")
    missing = unresolvable(list(dict.fromkeys(cited)), resolves)
    _require(not missing, f"report cites unresolvable ids: {missing}")

    members = row["members"].split(",") if row["members"] else []
    _, notes_by_id, tables = _tables(session)
    report = CommunityReport(
        lineage_id=lineage_id,
        level=0,
        title=title,
        summary=summary,
        rank=float(rank),
        findings=findings,
        cites=list(dict.fromkeys(cited)),
        generated_from=members,
        input_hash=community_input_hash(members, tables, notes_by_id),
        stale=False,
    )
    with session.operation({"tool": "write_community_report", "community": lineage_id}):
        session.store.write_report(report)
        session.resync()
    return {"lineage_id": lineage_id, "stale": False}


def write_entity_description(session: Session, slug: str, description: str) -> dict:
    # A merged-away name writes to the canonical page, never to the
    # orphaned one (docs/decisions/0001 §5) -- `get_entity` and `read`
    # already resolve the same way, and two tools must not disagree.
    slug = _canonical_slug(session, slugify(slug))
    try:
        page = session.store.read_entity_page(slug)
    except FrontMatterError as exc:
        # Unlike a report, an entity page carries `user:` overrides that a
        # blind overwrite could destroy -- there is nothing safe to
        # read-modify-write here, so this must surface as an actionable
        # error rather than silently replace the file (spec §10: Tier 2 is
        # never deleted by rebuild, and a malformed page is not "deleted",
        # so a human has to fix the front-matter by hand).
        raise ToolError(
            f"entity page for {slug!r} is malformed and cannot be updated: "
            f"{exc}; fix its front-matter by hand (see review_queue)"
        ) from exc
    if page is None:
        raise ToolError(f"no entity page for {slug!r}; run rebuild first")

    _require(bool(description.strip()), "an entity description cannot be empty")

    _, notes_by_id, tables = _tables(session)
    entity = tables.entities.get(slug)
    page.description = description
    page.stale = False
    if entity is not None:
        page.input_hash = entity_input_hash(entity, tables, notes_by_id)

    with session.operation({"tool": "write_entity_description", "entity": slug}):
        session.store.write_entity_page(page)
        session.resync()
    return {"slug": slug, "stale": False}


def rebuild_tool(session: Session, scope: str = "all") -> dict:
    if scope not in {"cache", "related_blocks", "all"}:
        raise ToolError(f"scope must be cache, related_blocks, or all; got {scope!r}")
    with session.operation({"tool": "rebuild", "scope": scope}):
        report = _rebuild(session, scope)
    return {
        "scope": scope,
        "notes_synced": report.synced,
        "pages_written": report.pages_written,
        "pages_marked_stale": report.pages_marked_stale,
        "reports_marked_stale": report.reports_marked_stale,
    }
