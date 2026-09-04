"""Spreadsheets via python-calamine (XLSX, XLS, XLSB, ODS); CSV and TSV via stdlib."""

from __future__ import annotations

import csv
import io
from pathlib import PurePath

from python_calamine import CalamineWorkbook

from mindpalace.ingest.ir import Block, Document, Locator, table_block
from mindpalace.ingest.parsers import register
from mindpalace.ingest.parsers.text import decode, make_source

TABLE_ROW_CAP = 2000


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _capped(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    if len(rows) <= TABLE_ROW_CAP + 1:
        return rows, 0
    return rows[: TABLE_ROW_CAP + 1], len(rows) - TABLE_ROW_CAP - 1


def parse_spreadsheet(name: str, data: bytes) -> Document:
    workbook = CalamineWorkbook.from_filelike(io.BytesIO(data))
    names = list(workbook.sheet_names)
    metadata: dict = {"sheet_names": names}
    truncated: dict[str, int] = {}
    blocks: list[Block] = []
    for index, sheet_name in enumerate(names, start=1):
        raw = workbook.get_sheet_by_name(sheet_name).to_python(skip_empty_area=True)
        rows = [[_cell(v) for v in row] for row in raw]
        rows, dropped = _capped(rows)
        if dropped:
            truncated[sheet_name] = dropped
        if rows:
            blocks.append(table_block(rows, locator=Locator("sheet", sheet_name, index)))
    if truncated:
        metadata["truncated_rows"] = truncated
    return Document(
        None,
        make_source(name, data, "application/vnd.ms-excel", "spreadsheet"),
        metadata,
        tuple(blocks),
    )


def _delimited(name: str, data: bytes, delimiter: str, mime: str, parser: str) -> Document:
    text = decode(data)
    rows = [list(r) for r in csv.reader(io.StringIO(text), delimiter=delimiter)]
    rows, dropped = _capped(rows)
    metadata: dict = {}
    if dropped:
        metadata["truncated_rows"] = {PurePath(name).name: dropped}
    block = table_block(rows, locator=Locator("sheet", PurePath(name).name, 1))
    return Document(None, make_source(name, data, mime, parser), metadata, (block,))


def parse_csv(name: str, data: bytes) -> Document:
    sample = decode(data[:4096])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;|\t").delimiter
    except csv.Error:
        delimiter = ","
    return _delimited(name, data, delimiter, "text/csv", "csv")


def parse_tsv(name: str, data: bytes) -> Document:
    return _delimited(name, data, "\t", "text/tab-separated-values", "tsv")


register("spreadsheet", parse_spreadsheet)
register("csv", parse_csv)
register("tsv", parse_tsv)
