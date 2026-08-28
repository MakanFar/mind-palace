from mindpalace.citations import extract_ids, unresolvable


def test_extract_ids_from_a_graphrag_style_citation():
    text = "The plateau is contested [Data: Entities (e_scaling-laws); Assertions (x_01ab)]."
    assert extract_ids(text) == ["e_scaling-laws", "x_01ab"]


def test_extract_ids_across_multiple_citations():
    text = "One [Data: Entities (e_a)]. Two [Data: Assertions (x_b, k_c)]."
    assert extract_ids(text) == ["e_a", "x_b", "k_c"]


def test_extract_ids_returns_empty_for_ungrounded_prose():
    assert extract_ids("A confident claim with no support.") == []


def test_extract_ids_ignores_the_more_marker():
    text = "[Data: Entities (e_a, e_b, +more)]"
    assert extract_ids(text) == ["e_a", "e_b"]


def test_unresolvable_reports_only_missing_ids():
    known = {"e_a", "x_b"}
    assert unresolvable(["e_a", "x_b", "e_ghost"], known.__contains__) == ["e_ghost"]


def test_unresolvable_is_empty_when_everything_resolves():
    assert unresolvable(["e_a"], {"e_a"}.__contains__) == []
