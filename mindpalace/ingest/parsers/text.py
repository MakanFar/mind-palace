"""Plain text with encoding detection, and Markdown (spec §Parsers)."""

from __future__ import annotations

import hashlib
import re

from charset_normalizer import from_bytes

from mindpalace.ingest.ir import Block, Document, SourceInfo
from mindpalace.ingest.parsers import PARSER_VERSION, register

_FENCE = re.compile(r"^```")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def make_source(name: str, data: bytes, mime: str, parser: str) -> SourceInfo:
    return SourceInfo(
        name=name,
        mime=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        parser=parser,
        parser_version=PARSER_VERSION,
    )


def decode(data: bytes) -> str:
    """UTF-8 first; otherwise charset-normalizer's best guess, which covers
    the GBK/Big5/Shift-JIS family a Chinese or Japanese source arrives in."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        best = from_bytes(data).best()
        return str(best) if best is not None else data.decode("utf-8", errors="replace")


def _paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


def parse_text(name: str, data: bytes) -> Document:
    text = decode(data)
    blocks = tuple(Block("paragraph", p) for p in _paragraphs(text))
    return Document(None, make_source(name, data, "text/plain", "text"), {}, blocks)


def parse_markdown(name: str, data: bytes) -> Document:
    """Headings, fences, and list items become blocks; everything else is a
    paragraph. The text is kept as written."""
    text = decode(data)
    blocks: list[Block] = []
    title: str | None = None
    lines = text.splitlines()
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Block("paragraph", "\n".join(para).strip()))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            j = i + 1
            while j < len(lines) and not _FENCE.match(lines[j]):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i + 1 : j])))
            i = j + 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            blocks.append(Block("heading", heading.group(2), level=level))
            if title is None and level == 1:
                title = heading.group(2)
            i += 1
            continue
        item = _LIST.match(line)
        if item:
            flush()
            blocks.append(Block("list_item", item.group(1).strip()))
            i += 1
            continue
        if not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return Document(
        title, make_source(name, data, "text/markdown", "markdown"), {}, tuple(blocks)
    )


register("text", parse_text)
register("markdown", parse_markdown)
