"""YAML front-matter parsing and rendering for vault markdown files."""

from __future__ import annotations

import yaml

DELIMITER = "---"


class FrontMatterError(ValueError):
    """Raised when a file's front-matter block is malformed."""


def parse(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != DELIMITER:
        return {}, text

    for cursor in range(1, len(lines)):
        if lines[cursor].strip() != DELIMITER:
            continue
        raw = "".join(lines[1:cursor])
        body = "".join(lines[cursor + 1 :])
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise FrontMatterError(f"invalid YAML front-matter: {exc}") from exc
        if not isinstance(data, dict):
            raise FrontMatterError("front-matter must be a mapping")
        return data, body.lstrip("\n")

    raise FrontMatterError("unterminated front-matter block")


def render(data: dict, body: str) -> str:
    raw = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
    return f"{DELIMITER}\n{raw}{DELIMITER}\n\n{body.rstrip()}\n"
