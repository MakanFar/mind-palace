"""Text units: the provenance anchor for everything extracted (spec §IR).

Offsets, not copies. A unit is a half-open span into the rendered markdown,
written into the capture's front-matter at ingest so ids stay stable even
when this chunker changes later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mindpalace.ingest.ir import (
    Block,
    Document,
    Locator,
    SourceInfo,
    locator_heading,
    render_block,
    to_markdown,
)

UNIT_BUDGET = 1200
UNIT_OVERLAP = 150
# Latin sentence ends need trailing whitespace ("3.5" is not an end); CJK
# full-width stops are ends on their own, since CJK text carries no spaces.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[。！？]")
_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class TextUnit:
    ordinal: int
    start: int
    end: int
    locator: Locator | None = None


@dataclass(frozen=True)
class _Item:
    """One block as laid out in the body. `start` includes the locator
    heading that `to_markdown` emitted just before it, if any."""

    start: int
    end: int
    block: Block


def _layout(document: Document) -> list[_Item]:
    """Mirror `to_markdown` exactly, tracking where each block lands."""
    items: list[_Item] = []
    cursor = 0
    current: Locator | None = None
    first = True
    for block in document.blocks:
        prefix = ""
        if block.locator is not None and block.locator != current:
            current = block.locator
            heading = locator_heading(current)
            if heading is not None:
                prefix = heading + _SEPARATOR
        if not first:
            cursor += len(_SEPARATOR)
        first = False
        start = cursor
        cursor += len(prefix) + len(render_block(block))
        items.append(_Item(start, cursor, block))
    return items


def _split_oversized(body: str, item: _Item) -> list[tuple[int, int]]:
    """Cut one block that exceeds the budget on its own.

    Paragraphs cut at sentence ends and the next piece starts UNIT_OVERLAP
    characters back; tables cut between rows (never inside one); anything
    else cuts at the last newline or space inside the budget.
    """
    start, end, block = item.start, item.end, item.block
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + UNIT_BUDGET, end)
        if limit == end:
            pieces.append((cursor, end))
            break
        window = body[cursor:limit]
        if block.kind == "table":
            cut = window.rfind("\n")
            cut = cursor + cut if cut > 0 else limit
            pieces.append((cursor, cut))
            cursor = cut
        elif block.kind == "paragraph":
            # The next piece starts UNIT_OVERLAP back from this cut, so the
            # cut must land more than UNIT_OVERLAP past the cursor or the
            # loop would rediscover the same sentence end forever and emit
            # one junk unit per character.
            ends = [m.end() for m in _SENTENCE_END.finditer(window)]
            cut = cursor + ends[-1] if ends else None
            if cut is None or cut - cursor <= UNIT_OVERLAP:
                space = window.rfind(" ")
                cut = cursor + space if space > UNIT_OVERLAP else limit
            pieces.append((cursor, cut))
            # Exactly UNIT_OVERLAP back, even mid-word: the overlap is a
            # retrieval aid, and a stable size is worth more than a clean
            # word boundary. `continue` skips the whitespace trim below.
            cursor = cut - UNIT_OVERLAP
            continue
        else:
            newline = window.rfind("\n")
            space = window.rfind(" ")
            at = newline if newline > 0 else space
            cut = cursor + at if at > 0 else limit
            pieces.append((cursor, cut))
            cursor = cut
        while cursor < end and body[cursor].isspace():
            cursor += 1
    return pieces


def _pack(body: str, items: list[_Item]) -> list[TextUnit]:
    units: list[TextUnit] = []
    open_start: int | None = None
    open_end = 0
    open_locator: Locator | None = None
    held: _Item | None = None  # a heading waiting for the block it introduces

    def close() -> None:
        nonlocal open_start
        if open_start is not None:
            units.append(TextUnit(len(units), open_start, open_end, open_locator))
            open_start = None

    def add(start: int, end: int, locator: Locator | None) -> None:
        nonlocal open_start, open_end, open_locator
        if open_start is not None and end - open_start > UNIT_BUDGET:
            close()
        if open_start is None:
            open_start, open_locator = start, locator
        open_end = end

    for item in items:
        if item.block.kind == "heading":
            if held is not None:
                # Two headings in a row: the earlier one has nothing of its
                # own to introduce, so it simply joins the current unit.
                add(held.start, held.end, held.block.locator)
            held = item
            continue
        lead_start = held.start if held is not None else item.start
        lead_locator = held.block.locator if held is not None else item.block.locator
        held = None
        if item.end - lead_start <= UNIT_BUDGET:
            add(lead_start, item.end, lead_locator)
            continue
        # Oversized: its pieces are units of their own.
        close()
        pieces = _split_oversized(body, _Item(lead_start, item.end, item.block))
        for piece_start, piece_end in pieces:
            units.append(TextUnit(len(units), piece_start, piece_end, lead_locator))
    if held is not None:
        add(held.start, held.end, held.block.locator)
    close()
    if units:
        last = units[-1]
        units[-1] = TextUnit(last.ordinal, last.start, len(body), last.locator)
    return units


def to_text_units(document: Document) -> list[TextUnit]:
    body = to_markdown(document)
    if not body:
        return []
    return _pack(body, _layout(document))


def _paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_text(body: str) -> list[tuple[int, int]]:
    """Unit spans for a bare string (a typed capture), packed exactly as a
    paragraph-only Document would be, then mapped back onto the original so
    the user's own text is stored untouched."""
    paragraphs = _paragraphs(body)
    if not paragraphs:
        return [(0, len(body))]
    document = Document(
        None,
        SourceInfo("", "text/plain", "", len(body), "text", 1),
        {},
        tuple(Block("paragraph", p) for p in paragraphs),
    )
    units = to_text_units(document)
    if len(units) <= 1:
        return [(0, len(body))]
    # Map rendered offsets back onto the original paragraph by paragraph.
    # Rendering only changes what sits *between* paragraphs (any run of
    # blank-ish lines becomes one "\n\n"), so each paragraph's text is
    # identical in both and a rendered offset inside paragraph i maps by
    # a constant shift; an offset in a separator maps to the next start.
    rendered_starts: list[int] = []
    original_starts: list[int] = []
    cursor_r = 0
    cursor_o = 0
    for paragraph in paragraphs:
        rendered_starts.append(cursor_r)
        at = body.find(paragraph, cursor_o)
        if at < 0:
            return [(0, len(body))]
        original_starts.append(at)
        cursor_r += len(paragraph) + len(_SEPARATOR)
        cursor_o = at + len(paragraph)

    def to_original(offset: int) -> int:
        for i in range(len(paragraphs) - 1, -1, -1):
            start = rendered_starts[i]
            if offset >= start:
                within = min(offset - start, len(paragraphs[i]))
                return original_starts[i] + within
        return 0

    spans = [(to_original(u.start), to_original(u.end)) for u in units]
    spans[-1] = (spans[-1][0], len(body))
    return spans
