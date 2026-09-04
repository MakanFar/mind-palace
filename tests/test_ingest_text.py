from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.text import decode, parse_markdown, parse_text
from mindpalace.ingest.sniff import detect


def test_detect_by_extension_then_by_magic_then_text():
    assert detect("a.PDF", b"") == "pdf"
    assert detect("a.docx", b"") == "docx"
    assert detect("a.xlsx", b"") == "spreadsheet"
    assert detect("a.ods", b"") == "spreadsheet"
    assert detect("a.tsv", b"") == "tsv"
    assert detect("a.htm", b"") == "html"
    assert detect("a.md", b"") == "markdown"
    assert detect("noext", b"%PDF-1.7 ...") == "pdf"
    assert detect("noext", b"<!doctype html><html>") == "html"
    assert detect("noext", b"just words") == "text"


def test_decode_handles_utf8_and_a_legacy_encoding():
    assert decode("héllo".encode("utf-8")) == "héllo"
    assert decode("中文测试内容，这是一段足够长的文本。".encode("gb18030")).startswith("中文")


def test_plain_text_becomes_paragraphs_with_no_title():
    doc = parse_text("t.txt", b"one line\n\nsecond para\nstill second\n")
    assert doc.title is None
    assert [b.kind for b in doc.blocks] == ["paragraph", "paragraph"]
    assert doc.blocks[1].text == "second para\nstill second"
    assert doc.source.parser == "text" and len(doc.source.sha256) == 64


def test_markdown_finds_headings_and_fences_and_keeps_text():
    md = b"# Title\n\nintro\n\n```py\nx=1\n```\n\n## Sub\n\n- a\n- b\n"
    doc = parse_markdown("t.md", md)
    assert doc.title == "Title"
    kinds = [(b.kind, b.level) for b in doc.blocks]
    assert kinds[0] == ("heading", 1) and ("code", None) in kinds and ("heading", 2) in kinds
    assert [b.text for b in doc.blocks if b.kind == "list_item"] == ["a", "b"]


def test_parse_bytes_falls_back_to_text_and_records_the_error(monkeypatch):
    from mindpalace.ingest import parsers

    def boom(name, data):
        raise RuntimeError("bad pdf")

    monkeypatch.setitem(parsers._REGISTRY, "pdf", boom)
    doc = parse_bytes("x.pdf", b"%PDF-1.4 not really")
    assert doc.source.parser == "text"
    assert doc.metadata["parser_error"].startswith("pdf: RuntimeError: bad pdf")
    assert doc.blocks


def test_parse_bytes_of_an_unregistered_format_is_text():
    doc = parse_bytes("weird.xyz", b"hello")
    assert doc.source.parser == "text" and doc.blocks[0].text == "hello"
