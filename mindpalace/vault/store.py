"""Tiered read/write access to the vault.

Tier 1 (captures, notes) is written once and never mutated. Tier 2 (entity
pages, community reports) is rewritten in place but never deleted by rebuild.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from mindpalace.atomic import atomic_write, cas_write
from mindpalace.frontmatter import parse, render
from mindpalace.models import (
    Capture,
    CommunityReport,
    EntityPage,
    Note,
    capture_from_markdown,
    capture_to_markdown,
    note_from_markdown,
    note_to_markdown,
)
from mindpalace.vault.paths import VaultPaths

RELATED_OPEN = "<!-- mindpalace:related -->"
RELATED_CLOSE = "<!-- /mindpalace:related -->"
GENERATED_BANNER = (
    "<!-- Machine-written. Edits below are replaced on regeneration; "
    "put durable changes under `user:` in the front-matter. -->"
)


class VaultStore:
    def __init__(self, paths: VaultPaths) -> None:
        self.paths = paths

    # ---- Tier 1 -------------------------------------------------------

    def write_capture(self, capture: Capture, created: datetime) -> Path:
        path = self.paths.capture_path(capture.id, created)
        cas_write(path, capture_to_markdown(capture), None)
        return path

    def read_capture(self, path: Path) -> Capture:
        return capture_from_markdown(path.read_text(encoding="utf-8"))

    def iter_captures(self) -> Iterator[Capture]:
        for path in sorted(self.paths.captures.glob("*.md")):
            yield self.read_capture(path)

    def write_note(self, note: Note, slug: str) -> Path:
        path = self.paths.note_path(note.id, slug)
        cas_write(path, note_to_markdown(note), None)
        return path

    def read_note(self, path: Path) -> Note:
        return note_from_markdown(path.read_text(encoding="utf-8"))

    def iter_notes(self) -> Iterator[Note]:
        for path in sorted(self.paths.notes.glob("*.md")):
            yield self.read_note(path)

    # ---- Tier 2 -------------------------------------------------------

    def write_entity_page(self, page: EntityPage) -> Path:
        data = {
            "id": f"e_{page.slug}",
            "type": page.type,
            "generated_from": page.generated_from,
            "input_hash": page.input_hash,
            "stale": page.stale,
        }
        if page.user:
            data["user"] = page.user
        related = "\n".join(f"- {line}" for line in page.related)
        body = (
            f"{GENERATED_BANNER}\n\n{page.description}\n\n"
            f"{RELATED_OPEN}\n{related}\n{RELATED_CLOSE}"
        )
        path = self.paths.entity_path(page.slug)
        atomic_write(path, render(data, body))
        return path

    def read_entity_page(self, slug: str) -> EntityPage | None:
        path = self.paths.entity_path(slug)
        if not path.exists():
            return None
        data, body = parse(path.read_text(encoding="utf-8"))
        description, related = _split_related(body)
        return EntityPage(
            slug=slug,
            type=data["type"],
            description=description,
            generated_from=data.get("generated_from", []),
            input_hash=data.get("input_hash", ""),
            stale=data.get("stale", True),
            user=data.get("user", {}),
            related=related,
        )

    def iter_entity_pages(self) -> Iterator[EntityPage]:
        for path in sorted(self.paths.entities.glob("*.md")):
            page = self.read_entity_page(path.stem)
            if page is not None:
                yield page

    def write_report(self, report: CommunityReport) -> Path:
        data = {
            "lineage_id": report.lineage_id,
            "level": report.level,
            "title": report.title,
            "rank": report.rank,
            "cites": report.cites,
            "generated_from": report.generated_from,
            "input_hash": report.input_hash,
            "stale": report.stale,
            "findings": report.findings,
        }
        path = self.paths.community_path(report.lineage_id)
        atomic_write(path, render(data, f"{GENERATED_BANNER}\n\n{report.summary}"))
        return path

    def read_report(self, lineage_id: str) -> CommunityReport | None:
        path = self.paths.community_path(lineage_id)
        if not path.exists():
            return None
        data, body = parse(path.read_text(encoding="utf-8"))
        return CommunityReport(
            lineage_id=data["lineage_id"],
            level=data["level"],
            title=data["title"],
            summary=_strip_banner(body),
            rank=data["rank"],
            findings=data.get("findings", []),
            cites=data.get("cites", []),
            generated_from=data.get("generated_from", []),
            input_hash=data.get("input_hash", ""),
            stale=data.get("stale", False),
        )

    def iter_reports(self) -> Iterator[CommunityReport]:
        for path in sorted(self.paths.communities.glob("*.md")):
            report = self.read_report(path.stem)
            if report is not None:
                yield report


def _strip_banner(body: str) -> str:
    return body.replace(GENERATED_BANNER, "", 1).strip()


def _split_related(body: str) -> tuple[str, list[str]]:
    text = _strip_banner(body)
    if RELATED_OPEN not in text:
        return text.strip(), []
    description, _, remainder = text.partition(RELATED_OPEN)
    block, _, _ = remainder.partition(RELATED_CLOSE)
    related = [
        line.removeprefix("- ").strip()
        for line in block.strip().splitlines()
        if line.strip()
    ]
    return description.strip(), related
