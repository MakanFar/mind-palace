"""Format detection: extension first, magic bytes second, text last (spec §Parsers)."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePath

FORMATS = ("pdf", "docx", "pptx", "spreadsheet", "csv", "tsv", "html", "markdown", "text")

_BY_EXTENSION = {
    "pdf": "pdf",
    "docx": "docx",
    "pptx": "pptx",
    "xlsx": "spreadsheet",
    "xlsm": "spreadsheet",
    "xls": "spreadsheet",
    "xlsb": "spreadsheet",
    "ods": "spreadsheet",
    "csv": "csv",
    "tsv": "tsv",
    "html": "html",
    "htm": "html",
    "xhtml": "html",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
}


def _zip_kind(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None
    if any(n.startswith("word/") for n in names):
        return "docx"
    if any(n.startswith("ppt/") for n in names):
        return "pptx"
    if any(n.startswith("xl/") for n in names):
        return "spreadsheet"
    if "mimetype" in names:
        return "spreadsheet"  # ODS declares itself in its first entry
    return None


def detect(name: str, data: bytes) -> str:
    ext = PurePath(name).suffix.lower().lstrip(".")
    if ext in _BY_EXTENSION:
        return _BY_EXTENSION[ext]
    head = data[:2048].lstrip()
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        return _zip_kind(data) or "text"
    lowered = head[:256].lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
        return "html"
    return "text"
