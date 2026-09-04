from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.pdf import parse_pdf


def pdf_bytes(pages: list[str], title: str | None = None) -> bytes:
    """A minimal two-object-per-page PDF with a text-layer content stream,
    built by hand so the test owns its fixture. pypdf reads it."""
    kids = []
    font_obj = 3
    page_objs = []
    for text in pages:
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        page_objs.append(content)
    n = 4
    body: list[tuple[int, bytes]] = []
    for content in page_objs:
        page_num, stream_num = n, n + 1
        kids.append(page_num)
        body.append(
            (
                page_num,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {stream_num} 0 R "
                f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode(),
            )
        )
        body.append(
            (
                stream_num,
                b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            )
        )
        n += 2
    info_num = n
    info = b"<< " + (f"/Title ({title}) ".encode() if title else b"") + b"/Author (Tester) >>"
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids ["
        + b" ".join(f"{k} 0 R".encode() for k in kids)
        + b"] /Count %d >>" % len(kids),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        info_num: info,
    }
    for num, obj in body:
        objs[num] = obj
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    count = max(objs) + 1
    out += f"xref\n0 {count}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, count):
        out += f"{offsets.get(num, 0):010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {count} /Root 1 0 R /Info {info_num} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_pdf_pages_become_page_locators_with_metadata():
    doc = parse_pdf("p.pdf", pdf_bytes(["Hello page one", "Second page text"], title="A Paper"))
    assert doc.title == "A Paper"
    assert doc.metadata["page_count"] == 2
    assert doc.metadata["author"] == "Tester"
    assert [b.locator.label for b in doc.blocks] == ["1", "2"]
    assert "Hello page one" in doc.blocks[0].text


def test_pdf_without_title_leaves_it_none():
    doc = parse_bytes("p.pdf", pdf_bytes(["x"]))
    assert doc.title is None and doc.source.parser == "pdf"


def test_a_pdf_with_no_text_layer_still_yields_a_document():
    doc = parse_pdf("p.pdf", pdf_bytes([""]))
    assert doc.metadata["page_count"] == 1
    assert doc.blocks == ()
