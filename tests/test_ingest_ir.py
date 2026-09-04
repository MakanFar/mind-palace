import pytest

from mindpalace.ingest.ir import (
    Block, Document, Locator, SourceInfo, table_block, to_markdown,
)


def source(name="a.txt", parser="text"):
    return SourceInfo(name=name, mime="text/plain", sha256="0" * 64, size=1, parser=parser, parser_version=1)


def test_block_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="kind"):
        Block(kind="footnote", text="x")
    with pytest.raises(ValueError, match="level"):
        Block(kind="heading", text="x", level=9)
    with pytest.raises(ValueError, match="locator"):
        Locator(kind="chapter", label="1", index=1)


def test_markdown_renders_headings_paragraphs_and_locator_headings():
    doc = Document(
        title="T",
        source=source(),
        metadata={},
        blocks=(
            Block("heading", "Intro", level=1, locator=Locator("page", "1", 1)),
            Block("paragraph", "First.", locator=Locator("page", "1", 1)),
            Block("paragraph", "Second.", locator=Locator("page", "2", 2)),
            Block("list_item", "an item", locator=Locator("page", "2", 2)),
            Block("code", "x = 1", locator=Locator("page", "2", 2)),
        ),
    )
    assert to_markdown(doc) == (
        "## Page 1\n\n# Intro\n\nFirst.\n\n## Page 2\n\nSecond.\n\n- an item\n\n```\nx = 1\n```\n"
    )


def test_markdown_emits_each_locator_heading_once_and_none_for_unlocated_blocks():
    doc = Document(title=None, source=source(), metadata={}, blocks=(
        Block("paragraph", "a"),
        Block("paragraph", "b", locator=Locator("sheet", "Q1", 1)),
        Block("paragraph", "c", locator=Locator("sheet", "Q1", 1)),
    ))
    assert to_markdown(doc) == "a\n\n## Sheet Q1\n\nb\n\nc\n"


def test_table_block_renders_gfm_and_escapes_pipes():
    block = table_block([["h1", "h|2"], ["1", "2"]])
    assert block.kind == "table"
    assert block.text == "| h1 | h\\|2 |\n| --- | --- |\n| 1 | 2 |"
    assert to_markdown(Document(None, source(), {}, (block,))) == block.text + "\n"
