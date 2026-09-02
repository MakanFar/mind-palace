"""Canonical on-disk layout for a Mind Palace vault."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mindpalace.ids import slugify

SUFFIX_LENGTH = 6
#: A note's file name carries the slug of its first line for human
#: orientation. An unbounded slug overflowed the filesystem's 255-byte name
#: limit on a note whose first line was a full paragraph, and the write
#: failed with ENAMETOOLONG after the capture was already saved.
MAX_NOTE_SLUG = 80


class VaultPaths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.mindpalace_md = self.root / "MINDPALACE.md"
        self.index_md = self.root / "index.md"
        self.captures = self.root / "captures"
        self.notes = self.root / "notes"
        self.entities = self.root / "entities"
        self.communities = self.root / "communities"
        self.graph_db = self.root / ".graph" / "mindpalace.db"
        self.decisions_log = self.root / ".mindpalace" / "decisions.jsonl"
        self.op_log = self.root / ".mindpalace" / "log.jsonl"
        self.vocabulary_log = self.root / ".mindpalace" / "vocabulary.jsonl"
        self.merges_log = self.root / ".mindpalace" / "merges.jsonl"
        self.lock = self.root / ".mindpalace" / "lock"

    def capture_path(self, capture_id: str, created: datetime) -> Path:
        suffix = capture_id[-SUFFIX_LENGTH:]
        return self.captures / f"{created:%Y-%m-%d-%H%M}-{suffix}.md"

    def note_path(self, note_id: str, slug: str) -> Path:
        return self.notes / f"{note_id}-{_truncate_slug(slugify(slug))}.md"

    def entity_path(self, slug: str) -> Path:
        return self.entities / f"{slugify(slug)}.md"

    def community_path(self, lineage_id: str) -> Path:
        """Keyed by lineage id alone. Including the title would mint a new file
        every time a report is retitled, leaving orphans behind."""
        return self.communities / f"{lineage_id}.md"

    def all_directories(self) -> list[Path]:
        return [
            self.captures,
            self.notes,
            self.entities,
            self.communities,
            self.graph_db.parent,
            self.op_log.parent,
        ]


def _truncate_slug(slug: str) -> str:
    """Cut to MAX_NOTE_SLUG at a word boundary where one exists."""
    if len(slug) <= MAX_NOTE_SLUG:
        return slug
    cut = slug[:MAX_NOTE_SLUG]
    head, _, _ = cut.rpartition("-")
    return (head or cut).rstrip("-")
