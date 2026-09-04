"""HTML via beautifulsoup4, with Utopia's content-root rule (spec §Parsers)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from mindpalace.ingest.ir import Block, Document, Locator, table_block
from mindpalace.ingest.parsers import register
from mindpalace.ingest.parsers.text import decode, make_source

_DROP = ("script", "style", "noscript", "nav", "header", "footer", "aside")
_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table")
_NESTING_TAGS = list(_BLOCK_TAGS)


def _root(soup: BeautifulSoup) -> Tag:
    """The first of main, [role=main], article; else body; else everything."""
    for selector in ("main", "[role=main]", "article"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    return soup.body or soup


def _table_rows(table: Tag) -> list[list[str]]:
    """Rows and cells that belong to this table directly; a nested table's
    contents stay inside the cell text of the row that holds it."""
    return [
        [
            cell.get_text(" ", strip=True)
            for cell in tr.find_all(["th", "td"])
            if cell.find_parent("table") is table
        ]
        for tr in table.find_all("tr")
        if tr.find_parent("table") is table
    ]


def parse_html(name: str, data: bytes) -> Document:
    soup = BeautifulSoup(decode(data), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    title = title or None
    root = _root(soup)
    for tag in root.find_all(_DROP):
        tag.decompose()

    blocks: list[Block] = []
    section: Locator | None = None
    heading_count = 0
    # find_all walks the tree in document order, so block order follows the page.
    for el in root.find_all(_BLOCK_TAGS):
        if el.find_parent("table") is not None:
            continue  # anything inside a table is rendered by the outermost table block
        # An element nested inside another block element (a <p> in an <li>,
        # a nested list, a table in a cell) is already covered by the
        # outermost one's text; emitting it too would duplicate content.
        if el.find_parent(_NESTING_TAGS) is not None:
            continue
        if el.name == "table":
            blocks.append(table_block(_table_rows(el), locator=section))
            continue
        text = el.get_text(" ", strip=True) if el.name != "pre" else el.get_text()
        if not text.strip():
            continue
        if el.name[0] == "h":
            heading_count += 1
            section = Locator("section", text, heading_count)
            blocks.append(Block("heading", text, level=int(el.name[1]), locator=section))
        elif el.name == "li":
            blocks.append(Block("list_item", text, locator=section))
        elif el.name == "pre":
            blocks.append(Block("code", text.strip("\n"), locator=section))
        else:
            blocks.append(Block("paragraph", text, locator=section))
    return Document(title, make_source(name, data, "text/html", "html"), {}, tuple(blocks))


register("html", parse_html)
