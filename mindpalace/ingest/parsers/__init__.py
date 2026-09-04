"""Dispatch by detected format; never raise (spec §Parsers).

A parser that fails does not lose the file: the plain-text fallback runs on
the same bytes and `metadata["parser_error"]` says what happened.
"""

from __future__ import annotations

from collections.abc import Callable

from mindpalace.ingest.ir import Document
from mindpalace.ingest.sniff import detect

PARSER_VERSION = 1

_REGISTRY: dict[str, Callable[[str, bytes], Document]] = {}


def register(format_key: str, fn: Callable[[str, bytes], Document]) -> None:
    _REGISTRY[format_key] = fn


def parse_bytes(name: str, data: bytes) -> Document:
    from mindpalace.ingest.parsers.text import parse_text

    fmt = detect(name, data)
    parser = _REGISTRY.get(fmt)
    if parser is None:
        return parse_text(name, data)
    try:
        return parser(name, data)
    except Exception as exc:  # noqa: BLE001 - surviving is the whole point
        fallback = parse_text(name, data)
        metadata = dict(fallback.metadata)
        metadata["parser_error"] = f"{fmt}: {type(exc).__name__}: {exc}"[:500]
        return Document(fallback.title, fallback.source, metadata, fallback.blocks)


# Each parser module registers itself on import. Add new ones here.
from mindpalace.ingest.parsers import text as _text  # noqa: E402,F401
