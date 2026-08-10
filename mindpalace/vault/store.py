"""Tiered read/write access to the vault.

Tier 1 (captures, notes) is written once and never mutated. Tier 2 (entity
pages, community reports) is rewritten in place but never deleted by rebuild.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from mindpalace.atomic import atomic_write, cas_write, content_hash
from mindpalace.frontmatter import FrontMatterError, parse, render
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
        # Validate that description doesn't contain marker lines
        for line in page.description.splitlines():
            if line.strip() == RELATED_OPEN:
                raise ValueError(f"Description contains marker line: {RELATED_OPEN}")
            if line.strip() == RELATED_CLOSE:
                raise ValueError(f"Description contains marker line: {RELATED_CLOSE}")

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
        # Anything a human hand-wrote below the machine-owned block must
        # survive regeneration -- only the description above it and the
        # block itself are ever machine-authored (spec §10).
        if page.trailing:
            body = f"{body}\n\n{page.trailing}"
        path = self.paths.entity_path(page.slug)
        content = render(data, body)
        if page.source_hash is not None:
            # This object came from `read_entity_page`: CAS-protect the
            # write so a concurrent Obsidian save is a detected conflict,
            # not a silently lost edit (spec §9.2). A page built fresh
            # (source_hash unset) was never read from disk first -- most
            # callers (tests seeding a vault, a brand-new page) intend an
            # unconditional write, so it falls through to `atomic_write`.
            cas_write(path, content, page.source_hash)
        else:
            atomic_write(path, content)
        return path

    def read_entity_page(self, slug: str) -> EntityPage | None:
        """Returns `None` only when the file does not exist. A file that
        exists but cannot be parsed (bad YAML, or valid YAML missing a
        required key) raises `FrontMatterError` -- deliberately, not
        silently treated as absent, since a rebuild reacting to "page is
        None" would create a brand-new page over the broken one and destroy
        whatever the user was in the middle of typing (spec §10: Tier 2 is
        never deleted by rebuild). Callers that must not let one bad page
        take down a whole listing (`iter_entity_pages`) catch this and skip.
        """
        path = self.paths.entity_path(slug)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        data, body = parse(raw)
        try:
            entity_type = data["type"]
        except KeyError as exc:
            raise FrontMatterError(
                f"{path.name}: missing required front-matter key 'type'"
            ) from exc
        description, related, trailing = _split_related(body)
        return EntityPage(
            slug=slug,
            type=entity_type,
            description=description,
            generated_from=data.get("generated_from", []),
            input_hash=data.get("input_hash", ""),
            stale=data.get("stale", True),
            user=data.get("user", {}),
            related=related,
            trailing=trailing,
            source_hash=content_hash(raw),
        )

    def iter_entity_pages(self) -> Iterator[EntityPage]:
        """Skips a page that fails to parse rather than raising -- one bad
        entity page must never take down every other listing built on this
        (spec §10). Callers that need to know *which* file was skipped, to
        record it as a `vault_issue`, do their own scan (see
        `mindpalace.index.sync._load_entity_pages`)."""
        for path in sorted(self.paths.entities.glob("*.md")):
            try:
                page = self.read_entity_page(path.stem)
            except FrontMatterError:
                continue
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
        content = render(data, f"{GENERATED_BANNER}\n\n{report.summary}")
        if report.source_hash is not None:
            cas_write(path, content, report.source_hash)
        else:
            atomic_write(path, content)
        return path

    def read_report(self, lineage_id: str) -> CommunityReport | None:
        """See `read_entity_page`'s docstring: `None` means absent, a
        malformed-but-present file raises."""
        path = self.paths.community_path(lineage_id)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        data, body = parse(raw)
        missing = [k for k in ("lineage_id", "level", "title", "rank") if k not in data]
        if missing:
            raise FrontMatterError(
                f"{path.name}: missing required front-matter key(s) {missing}"
            )
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
            source_hash=content_hash(raw),
        )

    def iter_reports(self) -> Iterator[CommunityReport]:
        """See `iter_entity_pages`: skips a report that fails to parse."""
        for path in sorted(self.paths.communities.glob("*.md")):
            try:
                report = self.read_report(path.stem)
            except FrontMatterError:
                continue
            if report is not None:
                yield report


def _strip_banner(body: str) -> str:
    return body.replace(GENERATED_BANNER, "", 1).strip()


def _split_related(body: str) -> tuple[str, list[str], str]:
    """Split a page body into the description, the related-block lines, and
    whatever a human wrote below the block. That trailing segment is Tier 2
    content exactly like the description above the block -- `write_entity_page`
    must re-emit it, not drop it (spec §10)."""
    text = _strip_banner(body)
    lines = text.splitlines()

    # Find indices of marker lines (must be complete lines, not substring matches)
    open_idx = None
    close_idx = None
    for i, line in enumerate(lines):
        if line.strip() == RELATED_OPEN:
            open_idx = i
        elif line.strip() == RELATED_CLOSE:
            close_idx = i

    # No markers found
    if open_idx is None:
        return text.strip(), [], ""

    # Markers found - extract description (before open marker)
    description = "\n".join(lines[:open_idx]).strip()

    # Extract related block (between markers) and whatever follows the close
    # marker.
    if close_idx is None:
        # Malformed - marker open but not closed
        related_lines = lines[open_idx + 1 :]
        trailing_lines: list[str] = []
    else:
        related_lines = lines[open_idx + 1 : close_idx]
        trailing_lines = lines[close_idx + 1 :]

    related = [
        line.removeprefix("- ").strip()
        for line in related_lines
        if line.strip()
    ]
    trailing = "\n".join(trailing_lines).strip()

    return description, related, trailing
