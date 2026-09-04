"""PDF text layer via pypdf; no OCR (spec §Parsers)."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from mindpalace.ingest.ir import Block, Document, Locator
from mindpalace.ingest.parsers import register
from mindpalace.ingest.parsers.text import make_source

_INFO_KEYS = {
    "/Title": "title",
    "/Author": "author",
    "/CreationDate": "created",
    "/ModDate": "modified",
}


def _clean(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def parse_pdf(name: str, data: bytes) -> Document:
    reader = PdfReader(io.BytesIO(data))
    metadata: dict = {"page_count": len(reader.pages)}
    title: str | None = None
    info = reader.metadata or {}
    for key, field in _INFO_KEYS.items():
        value = _clean(info.get(key))
        if value is None:
            continue
        if field == "title":
            title = value
        else:
            metadata[field] = value
    blocks: list[Block] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        locator = Locator("page", str(index), index)
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                blocks.append(Block("paragraph", para, locator=locator))
    return Document(
        title, make_source(name, data, "application/pdf", "pdf"), metadata, tuple(blocks)
    )


register("pdf", parse_pdf)
