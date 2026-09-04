import io
import zipfile

from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.tables import TABLE_ROW_CAP, parse_csv, parse_spreadsheet, parse_tsv


def xlsx_bytes(sheets: dict[str, list[list[str]]]) -> bytes:
    """Minimal OOXML workbook with inline strings, enough for calamine."""
    ct = ['<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    wb_sheets, wb_rels, files = [], [], {}
    for i, (sname, rows) in enumerate(sheets.items(), start=1):
        ct.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        wb_sheets.append(f'<sheet name="{sname}" sheetId="{i}" r:id="rId{i}"/>')
        wb_rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        xml_rows = []
        for r, row in enumerate(rows, start=1):
            cells = "".join(f'<c r="{chr(64 + c)}{r}" t="inlineStr"><is><t>{v}</t></is></c>' for c, v in enumerate(row, start=1))
            xml_rows.append(f'<row r="{r}">{cells}</row>')
        files[f"xl/worksheets/sheet{i}.xml"] = f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    ct.append("</Types>")
    files["[Content_Types].xml"] = "".join(ct)
    files["_rels/.rels"] = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    files["xl/workbook.xml"] = f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{"".join(wb_sheets)}</sheets></workbook>'
    files["xl/_rels/workbook.xml.rels"] = f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(wb_rels)}</Relationships>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    return buf.getvalue()


def test_each_sheet_is_a_table_block_with_a_sheet_locator():
    doc = parse_spreadsheet("w.xlsx", xlsx_bytes({"Q1": [["a", "b"], ["1", "2"]], "Notes": [["only"]]}))
    assert doc.metadata["sheet_names"] == ["Q1", "Notes"]
    assert [b.locator.label for b in doc.blocks] == ["Q1", "Notes"]
    assert doc.blocks[0].text == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert "truncated_rows" not in doc.metadata


def test_rows_are_capped_and_the_cap_is_recorded():
    rows = [["n"]] + [[str(i)] for i in range(TABLE_ROW_CAP + 50)]
    doc = parse_spreadsheet("w.xlsx", xlsx_bytes({"S": rows}))
    assert doc.blocks[0].text.count("\n") == TABLE_ROW_CAP + 1  # header + separator + cap rows
    assert doc.metadata["truncated_rows"] == {"S": 50}


def test_csv_sniffs_the_delimiter_and_tsv_uses_tab():
    doc = parse_csv("c.csv", b"a;b\n1;2\n")
    assert doc.blocks[0].text == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    doc = parse_tsv("t.tsv", b"a\tb\n1\t2\n")
    assert doc.blocks[0].text == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert doc.blocks[0].locator.kind == "sheet" and doc.blocks[0].locator.label == "t.tsv"


def test_dispatch_reaches_the_spreadsheet_parser_for_xlsx():
    assert parse_bytes("w.xlsx", xlsx_bytes({"S": [["x"]]})).source.parser == "spreadsheet"
