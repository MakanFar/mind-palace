import pytest

from mindpalace.frontmatter import FrontMatterError, parse, render


def test_parse_extracts_mapping_and_body():
    text = "---\nid: n_01\nauthor: llm\n---\n\nBody text here.\n"
    data, body = parse(text)
    assert data == {"id": "n_01", "author": "llm"}
    assert body == "Body text here.\n"


def test_parse_returns_empty_mapping_when_absent():
    data, body = parse("Just a body.\n")
    assert data == {}
    assert body == "Just a body.\n"


def test_parse_rejects_unterminated_block():
    with pytest.raises(FrontMatterError, match="unterminated"):
        parse("---\nid: n_01\n\nBody\n")


def test_parse_rejects_non_mapping():
    with pytest.raises(FrontMatterError, match="mapping"):
        parse("---\n- one\n- two\n---\n\nBody\n")


def test_render_round_trips():
    data = {"id": "n_01", "entities": [{"name": "scaling-laws"}]}
    text = render(data, "Body text.")
    parsed, body = parse(text)
    assert parsed == data
    assert body.strip() == "Body text."


def test_render_starts_with_delimiter():
    assert render({"id": "n_01"}, "Body").startswith("---\n")


def test_parse_rejects_falsy_non_mapping():
    """Regression: falsy non-mapping values (false) must be rejected, not silently converted to empty dict."""
    with pytest.raises(FrontMatterError, match="mapping"):
        parse("---\nfalse\n---\n\nBody\n")


def test_parse_treats_null_as_empty():
    """Null front-matter (empty/absent content) becomes empty dict."""
    data, body = parse("---\nnull\n---\n\nBody\n")
    assert data == {}
    assert body == "Body\n"


def test_render_round_trips_preserves_leading_blank_lines():
    """Regression: round-trip must preserve genuine leading blank lines in body."""
    body_with_leading_blank = "\nSecond paragraph"
    text = render({"id": "n_01"}, body_with_leading_blank)
    parsed, body = parse(text)
    assert body == "\nSecond paragraph\n"
