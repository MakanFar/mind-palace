from mindpalace.ingest.parsers import parse_bytes
from mindpalace.ingest.parsers.html import parse_html

PAGE = b"""<!doctype html><html><head><title>A Post</title><style>p{}</style></head>
<body><nav>skip me</nav>
<main>
  <h1>Heading</h1>
  <p>First <b>bold</b> para.</p>
  <ul><li>one</li><li>two</li></ul>
  <pre>code here</pre>
  <table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
  <script>alert(1)</script>
</main>
<footer>also skip</footer></body></html>"""


def test_html_uses_the_main_container_and_maps_tags_to_blocks():
    doc = parse_html("p.html", PAGE)
    assert doc.title == "A Post"
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "list_item", "list_item", "code", "table"]
    assert doc.blocks[1].text == "First bold para."
    assert doc.blocks[5].text.startswith("| a | b |")
    assert doc.blocks[1].locator.label == "Heading"
    text = " ".join(b.text for b in doc.blocks)
    assert "skip me" not in text and "alert" not in text


def test_html_without_main_falls_back_to_body():
    doc = parse_html("p.html", b"<html><body><p>only</p></body></html>")
    assert [b.text for b in doc.blocks] == ["only"] and doc.title is None


def test_sniffing_finds_html_without_extension():
    assert parse_bytes("page", PAGE).source.parser == "html"


def test_nested_block_elements_are_not_emitted_twice():
    doc = parse_html("p.html", b"<html><body><ul><li><p>alpha</p></li><li>outer<ul><li>inner</li></ul></li></ul></body></html>")
    assert [(b.kind, b.text) for b in doc.blocks] == [("list_item", "alpha"), ("list_item", "outer inner")]
