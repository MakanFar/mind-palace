"""The intermediate representation every parser produces (docs/decisions/0002 §IR).

A `Document` is a title the file itself declared, metadata the file itself
states, and an ordered run of blocks. Nothing here is invented: a parser that
cannot find a title leaves it None, and the renderer never fills one in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BLOCK_KINDS = frozenset({"heading", "paragraph", "list_item", "table", "code"})
LOCATOR_KINDS = frozenset({"page", "slide", "sheet", "section", "line"})
LOCATOR_TITLES = {
    "page": "Page",
    "slide": "Slide",
    "sheet": "Sheet",
    "section": "Section",
    "line": "Line",
}


@dataclass(frozen=True)
class Locator:
    """Where in the source a block came from: page 3, slide 2, sheet "Q1"."""

    kind: str
    label: str
    index: int

    def __post_init__(self) -> None:
        if self.kind not in LOCATOR_KINDS:
            raise ValueError(f"unknown locator kind {self.kind!r}")


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    level: int | None = None
    locator: Locator | None = None

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {self.kind!r}")
        if self.kind == "heading" and not (self.level is None or 1 <= self.level <= 6):
            raise ValueError(f"heading level must be 1-6, got {self.level!r}")


@dataclass(frozen=True)
class SourceInfo:
    name: str
    mime: str
    sha256: str
    size: int
    parser: str
    parser_version: int


@dataclass(frozen=True)
class Document:
    title: str | None
    source: SourceInfo
    metadata: dict = field(default_factory=dict)
    blocks: tuple[Block, ...] = ()


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def table_block(rows: list[list[str]], locator: Locator | None = None) -> Block:
    """A GFM table from rows; the first row is the header. Empty input is an
    empty table block so the caller can still attach a locator to it."""
    if not rows:
        return Block("table", "", locator=locator)
    width = max(len(r) for r in rows)
    padded = [[_cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return Block("table", "\n".join(lines), locator=locator)


def render_block(block: Block) -> str:
    """One block as it appears in the capture body. The chunker relies on
    this being the only rendering, so offsets it computes line up."""
    if block.kind == "heading":
        return f"{'#' * (block.level or 1)} {block.text}"
    if block.kind == "list_item":
        return f"- {block.text}"
    if block.kind == "code":
        return f"```\n{block.text}\n```"
    return block.text


#: Locator kinds that get a synthetic heading in the body. A `section`
#: locator names a heading that is already in the text, so rendering it
#: again would print every heading twice.
RENDERED_LOCATORS = frozenset({"page", "slide", "sheet", "line"})


def locator_heading(locator: Locator) -> str | None:
    if locator.kind not in RENDERED_LOCATORS:
        return None
    return f"## {LOCATOR_TITLES[locator.kind]} {locator.label}"


def to_markdown(document: Document) -> str:
    """The capture body. A locator becomes a level-2 heading the first time
    it is seen, the way Utopia writes "## Page 3", so a reader of the file
    sees where each piece came from."""
    parts: list[str] = []
    current: Locator | None = None
    for block in document.blocks:
        if block.locator is not None and block.locator != current:
            current = block.locator
            heading = locator_heading(current)
            if heading is not None:
                parts.append(heading)
        parts.append(render_block(block))
    return "\n\n".join(parts) + "\n" if parts else ""
