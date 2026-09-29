import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import (
    ToolError,
    get_entity,
    graph_stats,
    neighbors,
    read,
    decide,
    save_capture,
    search_global,
    search_local,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def populated(session):
    capture = save_capture(session, "The plateau is about data exhaustion.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply binds scaling, not architecture.",
        entities=[
            {"name": "scaling-laws", "type": "concept", "description": "Compute vs loss."},
            {"name": "data-exhaustion", "type": "concept", "description": "Running out."},
        ],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    return session, capture, note


def test_read_dispatches_on_prefix(populated):
    session, capture, note = populated
    assert "data exhaustion" in read(session, capture["id"])["text"]
    assert "Data supply" in read(session, note["id"])["text"]
    assert read(session, "e_scaling-laws")["kind"] == "entity"


def test_read_rejects_an_unknown_id(populated):
    session, _, _ = populated
    with pytest.raises(ToolError, match="not found"):
        read(session, "n_missing")


def test_neighbors_are_empty_until_confirmation(populated):
    session, _, _ = populated
    assert neighbors(session, "e_scaling-laws")["neighbours"] == []


def test_neighbors_appear_once_confirmed(populated):
    session, _, note = populated
    decide(session, note["relationship_assertions"][0]["id"], "confirm", "test")
    result = neighbors(session, "e_scaling-laws")
    assert result["neighbours"][0]["slug"] == "data-exhaustion"
    assert result["neighbours"][0]["type"] == "contradicts"


def test_neighbors_filter_by_edge_type(populated):
    session, _, note = populated
    decide(session, note["relationship_assertions"][0]["id"], "confirm", "test")
    assert neighbors(session, "e_scaling-laws", edge_types=["supports"])["neighbours"] == []


def test_get_entity_resolves_through_aliases(populated):
    session, _, _ = populated
    page = session.store.read_entity_page("scaling-laws")
    page.user = {"aliases": ["scaling law"]}
    session.store.write_entity_page(page)
    assert get_entity(session, "Scaling Law")["slug"] == "scaling-laws"


def test_get_entity_rejects_an_unknown_name(populated):
    session, _, _ = populated
    with pytest.raises(ToolError, match="no entity"):
        get_entity(session, "nonexistent")


def test_get_entity_raises_on_an_ambiguous_alias(populated):
    """Spec §8.5: two entities claiming the same alias is never silently
    resolved in favour of one -- an ambiguous lookup must be an error the
    assistant can relay, naming every candidate, not a coin flip."""
    session, _, _ = populated
    for slug in ("scaling-laws", "data-exhaustion"):
        page = session.store.read_entity_page(slug)
        page.user = {"aliases": ["shared-name"]}
        session.store.write_entity_page(page)

    with pytest.raises(ToolError, match="ambiguous") as excinfo:
        get_entity(session, "shared-name")
    assert "scaling-laws" in str(excinfo.value)
    assert "data-exhaustion" in str(excinfo.value)


def test_get_entity_exact_slug_wins_over_someone_elses_alias(populated):
    """An exact slug match must win outright over any alias match -- only
    alias-to-alias ambiguity is an error."""
    session, _, _ = populated
    page = session.store.read_entity_page("scaling-laws")
    page.user = {"aliases": ["data-exhaustion"]}
    session.store.write_entity_page(page)

    assert get_entity(session, "data-exhaustion")["slug"] == "data-exhaustion"


def test_graph_stats_reports_distance_to_the_threshold(populated):
    session, _, _ = populated
    stats = graph_stats(session)
    assert stats["clustering"]["active"] is False
    assert stats["clustering"]["threshold"] == 150
    assert stats["clustering"]["remaining"] == 150 - stats["entities"]


def test_graph_stats_names_the_active_embedder(populated):
    session, _, _ = populated
    stats = graph_stats(session)
    assert stats["embedder"]["model_id"] == "stub-64"
    assert stats["embedder"]["sends_data_off_machine"] is False


def test_graph_stats_counts_orphans(populated):
    session, _, _ = populated
    assert graph_stats(session)["orphans"] == 2  # nothing confirmed yet


def test_search_local_wraps_retrieval(populated):
    # SQLite FTS5's bm25() uses the unsmoothed idf = ln((N - df + 0.5) /
    # (df + 0.5)); with only the two `populated` documents and one matching,
    # idf is exactly 0 and the match scores ~0 regardless of match quality --
    # a degenerate corpus, not a degenerate query (see the identical note in
    # tests/test_retrieve.py::test_local_search_returns_lexical_matches,
    # which is where this fixture's brief-supplied original version was
    # copied from without the distractors that make it exercise anything).
    # A handful of unrelated captures keeps idf non-zero so this test
    # actually exercises search_local's evidence gate instead of always
    # tripping the abstain path.
    session, _, _ = populated
    for text in (
        "sourdough starter hydration",
        "cold front moving through tonight",
        "tomatoes need staking this week",
        "the train was delayed again",
        "simmer the stock for two hours",
        "remember to renew the passport",
        "the new album drops on friday",
        "the bike chain keeps slipping",
        "book the ferry crossing early",
    ):
        save_capture(session, text)

    result = search_local(session, "data exhaustion")
    assert result["hits"]


def test_search_global_wraps_retrieval(populated):
    """The brief lists `search_global` among the interfaces this task
    produces but did not include a test for it (unlike every other
    function here). `mindpalace.retrieve.global_search` already has its
    own dedicated test module; this only needs to confirm the tools.py
    wrapper passes the session's pieces through correctly."""
    session, _, _ = populated
    result = search_global(session, "data exhaustion")
    assert result["available"] is False
    assert "No communities exist yet" in result["note"]


def test_read_returns_a_text_unit_and_search_finds_it(session):
    from datetime import UTC, datetime

    from mindpalace.models import Capture
    from mindpalace.tools import graph_stats, read, search_local

    from mindpalace.tools import save_capture

    # BM25's IDF is zero on a two-document corpus (see the retrieval tests),
    # so give the index enough unrelated material for a real match to clear
    # the abstain floor.
    for i in range(10):
        save_capture(session, f"unrelated filler number {i} about gardening and trains")
    body = "## Page 1\n\nsourdough hydration ratios\n\n## Page 2\n\nzeta"
    cut = body.index("## Page 2")
    session.store.write_capture(
        Capture("c_01", "2026-09-04T00:00:00Z", "file", None, body, units=((0, cut), (cut, len(body)))),
        datetime(2026, 9, 4, tzinfo=UTC),
    )
    session.resync()
    unit = read(session, "u_01_0000")
    assert unit["kind"] == "text_unit" and unit["capture"] == "c_01" and unit["locator"] == "Page 1"
    assert "sourdough" in unit["text"]
    hits = search_local(session, "sourdough hydration", k=3)["hits"]
    assert hits and hits[0]["id"] == "u_01_0000" and hits[0]["kind"] == "unit"
    stats = graph_stats(session)
    assert stats["text_units"] == 12 and stats["attachments"] == 0
