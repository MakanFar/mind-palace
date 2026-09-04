"""Identifier generation, kind dispatch, and name normalisation."""

from __future__ import annotations

import re
import unicodedata

from ulid import ULID

PREFIXES: dict[str, str] = {
    "c_": "capture",
    "n_": "note",
    "x_": "relationship_assertion",
    "k_": "claim_assertion",
    "e_": "entity",
    "g_": "community",
    "u_": "text_unit",
    "op_": "operation",
}

# Longest prefix first so "op_" is never shadowed by a single-letter prefix.
_ORDERED_PREFIXES = sorted(PREFIXES, key=len, reverse=True)


class UnknownIdError(ValueError):
    """Raised for an unregistered prefix or an unrecognisable identifier."""


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise UnknownIdError(f"unknown id prefix {prefix!r}")
    return f"{prefix}{ULID()}"


def id_kind(identifier: str) -> str:
    for prefix in _ORDERED_PREFIXES:
        if identifier.startswith(prefix):
            return PREFIXES[prefix]
    raise UnknownIdError(f"unrecognised identifier {identifier!r}")


def slugify(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    hyphenated = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower().strip())
    return hyphenated.strip("-")


def entity_id(name: str) -> str:
    return f"e_{slugify(name)}"


def aggregate_key(
    source: str, edge_type: str, target: str, *, symmetric: bool
) -> str:
    left, right = slugify(source), slugify(target)
    if symmetric:
        left, right = sorted((left, right))
    return f"r:{left}|{edge_type}|{right}"


def unit_id(capture_id: str, ordinal: int) -> str:
    """`u_<capture ulid>_<ordinal>` (docs/decisions/0002 §Storage). Derived,
    never minted: the same capture and ordinal always name the same unit."""
    return f"u_{capture_id.removeprefix('c_')}_{ordinal:04d}"


def unit_parent(identifier: str) -> tuple[str, int]:
    body = identifier.removeprefix("u_")
    ulid, _, ordinal = body.rpartition("_")
    return f"c_{ulid}", int(ordinal)
