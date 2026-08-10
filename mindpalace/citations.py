"""Extraction and validation of inline data citations.

Prose that cannot be checked is prose that cannot be repaired, so every
generated artifact carries both inline citations and a structural `cites` list.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

CITATION_PATTERN = re.compile(r"\[Data:([^\]]*)\]")
ID_PATTERN = re.compile(r"\b((?:e|x|k|n|c|g)_[A-Za-z0-9\-]+)\b")


class CitationError(ValueError):
    """Raised when a generated artifact cites something that does not exist."""


def extract_ids(text: str) -> list[str]:
    found: list[str] = []
    for block in CITATION_PATTERN.findall(text):
        for identifier in ID_PATTERN.findall(block):
            if identifier not in found:
                found.append(identifier)
    return found


def unresolvable(
    cites: Iterable[str], resolver: Callable[[str], bool]
) -> list[str]:
    return [identifier for identifier in cites if not resolver(identifier)]
