import io
import zipfile

from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.office import parse_docx, parse_pptx

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'


def zipped(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in entries.items():
            z.writestr(name, text)
    return buf.getvalue()


def docx_bytes() -> bytes:
    document = f"""<w:document {W}><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>
      <w:p><w:r><w:t>First </w:t></w:r><w:r><w:t>sentence.</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>h1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>h2</w:t></w:r></w:p></w:tc></w:tr>
             <w:tr><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Method</w:t></w:r></w:p>
      <w:p><w:r><w:t>Body.</w:t></w:r></w:p>
    </w:body></w:document>"""
    core = """<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>My Doc</dc:title><dc:creator>Ann</dc:creator></cp:coreProperties>"""
    return zipped({"word/document.xml": document, "docProps/core.xml": core})


def pptx_bytes() -> bytes:
    def slide(title, body):
        return f"""<p:sld {A}><p:cSld><p:spTree>
          <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
          <p:sp><p:txBody><a:p><a:r><a:t>{body}</a:t></a:r></a:p></p:txBody></p:sp>
        </p:spTree></p:cSld></p:sld>"""
    return zipped({
        "ppt/slides/slide10.xml": slide("Ten", "tenth body"),
        "ppt/slides/slide2.xml": slide("Two", "second body"),
        "ppt/slides/slide1.xml": slide("One", "first body"),
    })


def test_docx_headings_paragraphs_tables_and_core_metadata():
    doc = parse_docx("d.docx", docx_bytes())
    assert doc.title == "My Doc" and doc.metadata["author"] == "Ann"
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "table", "heading", "paragraph"]
    assert doc.blocks[1].text == "First sentence."
    assert doc.blocks[2].text.startswith("| h1 | h2 |")
    assert doc.blocks[1].locator.kind == "section" and doc.blocks[1].locator.label == "Intro"
    assert doc.blocks[4].locator.label == "Method" and doc.blocks[4].locator.index == 2


def test_pptx_slides_come_in_numeric_order_with_title_headings():
    doc = parse_pptx("s.pptx", pptx_bytes())
    assert doc.metadata["slide_count"] == 3
    labels = [b.locator.label for b in doc.blocks]
    assert labels == ["1", "1", "2", "2", "10", "10"]
    assert doc.blocks[0].kind == "heading" and doc.blocks[0].text == "One"
    assert doc.blocks[5].text == "tenth body"


def test_sniffing_routes_zip_without_extension():
    assert parse_bytes("blob", docx_bytes()).source.parser == "docx"
    assert parse_bytes("blob", pptx_bytes()).source.parser == "pptx"
