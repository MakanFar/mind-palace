"""DOCX and PPTX by unzipping the XML, as Utopia does; no library needed."""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile

from mindpalace.ingest.ir import Block, Document, Locator, table_block
from mindpalace.ingest.parsers import register
from mindpalace.ingest.parsers.text import make_source

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
DC = "{http://purl.org/dc/elements/1.1/}"
DCTERMS = "{http://purl.org/dc/terms/}"

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

_HEADING_STYLE = re.compile(r"^(?:Heading|heading)\s*(\d)$")
_SLIDE_ENTRY = re.compile(r"ppt/slides/slide(\d+)\.xml")


def _entry(archive: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        return archive.read(name)
    except KeyError:
        return None


def _core_metadata(archive: zipfile.ZipFile) -> tuple[str | None, dict]:
    """Title and the author/created/modified fields from docProps/core.xml,
    when the package has one."""
    raw = _entry(archive, "docProps/core.xml")
    if raw is None:
        return None, {}
    root = ET.fromstring(raw)
    title = (root.findtext(f"{DC}title") or "").strip() or None
    metadata: dict = {}
    for tag, key in (
        (f"{DC}creator", "author"),
        (f"{DCTERMS}created", "created"),
        (f"{DCTERMS}modified", "modified"),
    ):
        value = (root.findtext(tag) or "").strip()
        if value:
            metadata[key] = value
    return title, metadata


def _para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{W}t")).strip()


def _heading_level(p: ET.Element) -> int | None:
    style = p.find(f"{W}pPr/{W}pStyle")
    style_val = style.get(f"{W}val") if style is not None else None
    if not style_val:
        return None
    m = _HEADING_STYLE.match(style_val)
    if m:
        return max(1, min(int(m.group(1)), 6))
    if style_val == "Title":
        return 1
    return None


def parse_docx(name: str, data: bytes) -> Document:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        title, metadata = _core_metadata(archive)
    body = root.find(f"{W}body")
    blocks: list[Block] = []
    section: Locator | None = None
    heading_count = 0
    for child in list(body) if body is not None else []:
        if child.tag == f"{W}p":
            text = _para_text(child)
            if not text:
                continue
            level = _heading_level(child)
            if level is not None:
                heading_count += 1
                section = Locator("section", text, heading_count)
                blocks.append(Block("heading", text, level=level, locator=section))
                if title is None and level == 1:
                    title = text
            else:
                blocks.append(Block("paragraph", text, locator=section))
        elif child.tag == f"{W}tbl":
            rows = [
                [_para_text(cell) for cell in tr.iter(f"{W}tc")]
                for tr in child.iter(f"{W}tr")
            ]
            blocks.append(table_block(rows, locator=section))
    return Document(title, make_source(name, data, DOCX_MIME, "docx"), metadata, tuple(blocks))


def _slide_number(entry: str) -> int | None:
    m = _SLIDE_ENTRY.fullmatch(entry)
    return int(m.group(1)) if m else None


def _shape_blocks(shape: ET.Element, locator: Locator) -> list[Block]:
    ph = shape.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
    is_title = ph is not None and ph.get("type") in {"title", "ctrTitle"}
    blocks: list[Block] = []
    for para in shape.iter(f"{A}p"):
        text = "".join(t.text or "" for t in para.iter(f"{A}t")).strip()
        if not text:
            continue
        if is_title and not blocks:
            blocks.append(Block("heading", text, level=2, locator=locator))
        else:
            blocks.append(Block("paragraph", text, locator=locator))
    return blocks


def parse_pptx(name: str, data: bytes) -> Document:
    blocks: list[Block] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        slides = sorted(
            ((n, e) for e in archive.namelist() if (n := _slide_number(e)) is not None),
            key=lambda pair: pair[0],
        )
        title, metadata = _core_metadata(archive)
        metadata["slide_count"] = len(slides)
        for ordinal, (number, entry) in enumerate(slides, start=1):
            root = ET.fromstring(archive.read(entry))
            locator = Locator("slide", str(number), ordinal)
            for shape in root.iter(f"{P}sp"):
                blocks.extend(_shape_blocks(shape, locator))
    return Document(title, make_source(name, data, PPTX_MIME, "pptx"), metadata, tuple(blocks))


register("docx", parse_docx)
register("pptx", parse_pptx)
