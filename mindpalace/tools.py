"""Tool implementations as plain functions over a Session.

Keeping these free of MCP types makes them directly testable; server.py does
nothing but wire them to the protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mindpalace.citations import extract_ids, unresolvable
from mindpalace.cluster import Community, match_lineages, partition, should_cluster
from mindpalace.frontmatter import FrontMatterError
from mindpalace.ids import UnknownIdError, id_kind, new_id, slugify
from mindpalace.models import (
    Capture,
    ClaimAssertion,
    CommunityReport,
    EntityInstance,
    Note,
    RelationshipAssertion,
)
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
        try:
            page = session.store.read_entity_page(row["slug"])
        except FrontMatterError:
            # A malformed page must not break every other entity's listing
            # here (spec §10); it is already surfaced via vault_issues.
            page = None
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


def _tables(session: Session):
    """Fold the current source set. Several tools need it for evidence hashing.

    Quarantine-aware: a note `fold` rejects (unknown edge type, duplicate
    assertion id) must not take `cluster`, `write_entity_description`, or
    `write_community_report` down with it (spec §10). See
    `mindpalace.index.sync.fold_notes_with_quarantine`.
    """
    notes, notes_by_id, tables, _issues = fold_notes_with_quarantine(
        session.store, session.config, session.statuses()
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
            page = session.store.read_entity_page(identifier.removeprefix("e_"))
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
    slug = slugify(name)
    row = session.conn.execute(
        "SELECT slug FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    if row is not None:
        return row["slug"]
    matches: list[str] = []
    for page in session.store.iter_entity_pages():
        aliases = {slugify(alias) for alias in page.user.get("aliases", [])}
        if slug in aliases:
            matches.append(page.slug)
    if len(matches) > 1:
        candidates = sorted(set(matches))
        raise ToolError(
            f"alias {name!r} is ambiguous: claimed by more than one entity "
            f"{candidates}; aliases do not merge identities"
        )
    return matches[0] if matches else None


def get_entity(session: Session, name: str) -> dict:
    slug = _resolve_slug(session, name)
    if slug is None:
        raise ToolError(f"no entity matching {name!r}")

    try:
        page = session.store.read_entity_page(slug)
    except FrontMatterError:
        # A malformed page must not stop `get_entity` from returning what
        # the graph still knows (type, rank, neighbours) -- it degrades to
        # the same shape as "no page written yet" (spec §10).
        page = None
    row = session.conn.execute(
        "SELECT type, rank FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    claims = [
        dict(record)
        for record in session.conn.execute(
            "SELECT id, text, status FROM claims WHERE subject = ?", (slug,)
        )
    ]
    return {
        "slug": slug,
        "type": row["type"] if row else (page.type if page else "unknown"),
        "rank": row["rank"] if row else 0,
        "description": page.description if page else "",
        "stale": page.stale if page else True,
        "user": page.user if page else {},
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
    _require(
        type in session.config.edge_types,
        f"unknown edge type {type!r}; choose from {sorted(session.config.edge_types)}",
    )
    _require(
        isinstance(strength, int) and strength in STRENGTH_RANGE,
        f"strength must be an integer 1-10, got {strength!r}",
    )
    _require(bool(description.strip()), "a proposed relationship needs a rationale")

    assertion = RelationshipAssertion(
        id=new_id("x_"),
        source=_require_slug(source, "relationship source"),
        target=_require_slug(target, "relationship target"),
        type=type,
        strength=int(strength),
        description=description,
    )
    note = Note(
        id=new_id("n_"),
        derived_from=None,
        created=_timestamp(_now()),
        author="user",
        body=f"Relationship proposed outside extraction: {description}",
        relationship_assertions=(assertion,),
    )
    with session.operation({"tool": "propose_relationship", "note": note.id}):
        session.store.write_note(note, f"link-{assertion.source}-{assertion.target}")
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )
    return {
        "id": assertion.id,
        "note": note.id,
        "pair": f"{assertion.source}|{assertion.type}|{assertion.target}",
        "status": "proposed",
    }


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
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )

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
            "pair": f"{row['source']}|{row['type']}|{row['target']}",
            "strength": row["strength"],
            "why": row["description"],
            "note": row["note_id"],
            "source_snippet": _endpoint_snippet(session, row["source"]),
            "target_snippet": _endpoint_snippet(session, row["target"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, source, target, type, strength, description "
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
            "note": row["note_id"],
            "subject_snippet": _endpoint_snippet(session, row["subject"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, subject, text FROM claims "
            "WHERE status = 'proposed' ORDER BY id LIMIT ?",
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

    return {"proposals": proposals, "vault_issues": issues}


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
    slug = slugify(slug)
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
        report = rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
            scope=scope,
        )
    return {
        "scope": scope,
        "notes_synced": report.synced,
        "pages_written": report.pages_written,
        "pages_marked_stale": report.pages_marked_stale,
        "reports_marked_stale": report.reports_marked_stale,
    }
