from mindpalace.ingest.ir import Block, Document, Locator, SourceInfo, table_block, to_markdown
from mindpalace.ingest.units import UNIT_BUDGET, UNIT_OVERLAP, split_text, to_text_units


def doc(*blocks):
    src = SourceInfo("a", "text/plain", "0" * 64, 1, "text", 1)
    return Document(None, src, {}, tuple(blocks))


def test_short_document_is_one_unit_covering_the_whole_body():
    d = doc(Block("paragraph", "hello"), Block("paragraph", "world"))
    body = to_markdown(d)
    [unit] = to_text_units(d)
    assert (unit.ordinal, unit.start, unit.end) == (0, 0, len(body))
    assert body[unit.start:unit.end] == body


def test_blocks_are_packed_whole_until_the_budget():
    paragraphs = [Block("paragraph", f"p{i} " + "x" * 300) for i in range(8)]
    d = doc(*paragraphs)
    body = to_markdown(d)
    units = to_text_units(d)
    assert len(units) >= 2
    for u in units:
        assert u.end - u.start <= UNIT_BUDGET + UNIT_OVERLAP
        # every unit starts at a block boundary here: no paragraph is oversized
        assert body[u.start:u.start + 1] == "p"
    assert units[-1].end == len(body)


def test_oversized_paragraph_splits_at_sentence_ends_with_overlap():
    text = " ".join(f"Sentence number {i} ends here." for i in range(120))
    d = doc(Block("paragraph", text))
    body = to_markdown(d)
    units = to_text_units(d)
    assert len(units) > 1
    for a, b in zip(units, units[1:]):
        assert body[a.end - 1] == "."
        assert a.end - b.start == UNIT_OVERLAP or b.start == a.end
    assert units[-1].end == len(body)


def test_a_table_is_never_split_inside_a_row():
    rows = [["h"]] + [[f"cell {i} " + "y" * 60] for i in range(60)]
    d = doc(table_block(rows))
    body = to_markdown(d)
    for u in to_text_units(d):
        assert body[u.start:u.end].strip().startswith("|")
        assert body[u.start:u.end].rstrip().endswith("|")


def test_heading_travels_with_the_block_that_follows_it():
    filler = Block("paragraph", "z" * (UNIT_BUDGET - 10))
    d = doc(filler, Block("heading", "Section B", level=2), Block("paragraph", "body of B"))
    body = to_markdown(d)
    units = to_text_units(d)
    assert len(units) == 2
    assert body[units[1].start:units[1].end].startswith("## Section B")


def test_units_carry_the_locator_of_their_first_block():
    d = doc(Block("paragraph", "a", locator=Locator("page", "1", 1)),
            Block("paragraph", "b" * UNIT_BUDGET, locator=Locator("page", "2", 2)))
    units = to_text_units(d)
    assert units[0].locator == Locator("page", "1", 1)
    assert units[-1].locator == Locator("page", "2", 2)


def test_split_text_matches_a_paragraph_only_document():
    text = "one\n\ntwo\n\nthree"
    assert split_text(text) == [(0, len(text))]
    long = "\n\n".join("w" * 500 for _ in range(5))
    spans = split_text(long)
    assert spans[0][0] == 0 and spans[-1][1] == len(long) and len(spans) >= 2
