"""The window's data (docs/decisions/0003 §Part 1). Tier 3: rewritten on every sync.

Everything the Obsidian view shows and nothing it must compute. Ids everywhere
so the panel can link to notes, captures, and units without a second read.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from mindpalace.atomic import atomic_write
from mindpalace.config import Config
from mindpalace.graph.fold import GraphTables
from mindpalace.models import Capture, CommunityReport, EntityPage, Note

VERSION = 1


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def build_graph_export(
    tables: GraphTables,
    config: Config,
    captures: list[Capture],
    capture_paths: dict[str, str],
    notes_by_id: dict[str, Note],
    note_paths: dict[str, str],
    units: Iterable,
    communities: list[dict],
    vocabulary: list[tuple[str, str, int, str, str]],
    entity_pages: list[EntityPage],
    reports: list[CommunityReport],
    generated_at: str,
) -> dict:
    pages = {page.slug: page for page in entity_pages}
    report_by_lineage = {r.lineage_id: r for r in reports}

    entities = [
        {
            "slug": e.slug,
            "type": e.type,
            "rank": e.rank,
            "merged_from": list(e.merged_from),
            "description": _first_line(pages[e.slug].description) if e.slug in pages else "",
            "stale": pages[e.slug].stale if e.slug in pages else True,
            "note_ids": list(e.note_ids),
            "text_unit_ids": list(e.text_unit_ids),
        }
        for e in tables.entities.values()
    ]

    def assertion_row(a) -> dict:
        return {
            "id": a.id,
            "note_id": a.note_id,
            "status": a.status,
            "strength": a.strength,
            "description": a.description,
            "direction_corrected": a.direction_corrected,
            "text_unit_ids": list(a.text_unit_ids),
        }

    edges = []
    for agg in tables.aggregates.values():
        members = [tables.assertions[i] for i in agg.assertion_ids if i in tables.assertions]
        edges.append(
            {
                "key": agg.key,
                "source": agg.source,
                "target": agg.target,
                "type": agg.type,
                "proposed_type": None,
                "directed": not config.is_symmetric(agg.type),
                "weight": agg.weight,
                "traversable": agg.traversable,
                "assertions": [assertion_row(a) for a in members],
            }
        )
    # Untyped assertions form no aggregate (0001 §1); the window draws them
    # dotted between their endpoints from this list.
    untyped = [
        {
            "id": a.id,
            "source": a.source,
            "target": a.target,
            "proposed_type": a.proposed_type,
            "status": a.status,
            "note_id": a.note_id,
            "description": a.description,
            "text_unit_ids": list(a.text_unit_ids),
        }
        for a in tables.assertions.values()
        if a.type is None
    ]
    claims = [
        {
            "id": c.id,
            "subject": c.subject,
            "text": c.text,
            "status": c.status,
            "valid_from": c.valid_from,
            "valid_to": c.valid_to,
            "supersedes": c.supersedes,
            "note_id": c.note_id,
            "text_unit_ids": list(c.text_unit_ids),
        }
        for c in tables.claims.values()
    ]
    community_rows = []
    for c in communities:
        report = report_by_lineage.get(c["lineage_id"])
        community_rows.append(
            {
                "lineage_id": c["lineage_id"],
                "level": c["level"],
                "parent": c["parent"],
                "members": c["members"],
                "title": report.title if report else None,
                "stale": report.stale if report else None,
            }
        )
    return {
        "version": VERSION,
        "generated_at": generated_at,
        "entities": entities,
        "edges": edges,
        "untyped": untyped,
        "claims": claims,
        "communities": community_rows,
        "vocabulary": [
            {"kind": kind, "proposed": proposed, "count": count, "example": example}
            for kind, proposed, count, example, _ids in vocabulary
        ],
        "captures": {
            c.id: {"path": capture_paths.get(c.id, ""), "title": c.title or _first_line(c.text)}
            for c in captures
        },
        "notes": {note_id: {"path": note_paths.get(note_id, "")} for note_id in notes_by_id},
        "units": {u.id: {"capture_id": u.capture_id, "locator": u.locator} for u in units},
    }


def write_graph_export(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(payload, ensure_ascii=False))
