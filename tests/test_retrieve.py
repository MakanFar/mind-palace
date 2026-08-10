import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db, vectors
from mindpalace.retrieve import Hit, local_search, passes_evidence_gate, rrf


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={"abstain": "Nothing in the vault is relevant to this query."},
    )


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "cache.db")
    db.create_schema(connection)
    return connection


def index_doc(conn, embedder, doc_id, kind, title, text):
    conn.execute(
        "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
        (doc_id, kind, title, text),
    )
    conn.commit()
    vectors.store(conn, doc_id, kind, embedder.model_id, embedder.embed([text])[0])


def hit(doc_id, *, bm25=0.0, cosine=0.0):
    return Hit(
        id=doc_id, kind="note", title="t", snippet="s", bm25=bm25, cosine=cosine, score=1.0
    )


def test_rrf_prefers_documents_appearing_in_both_lists():
    # "a" is present only in the first list; "b" is present in both. RRF sums
    # a document's 1/(k+rank) contribution from every list it appears in, so
    # "b" accumulates two terms against "a"'s one and wins even though "a"
    # holds the better individual rank in the list they share.
    fused = rrf([["a", "b"], ["b"]])
    assert fused[0][0] == "b"


def test_rrf_respects_rank_order_within_a_single_list():
    assert [doc for doc, _ in rrf([["a", "b", "c"]])] == ["a", "b", "c"]


def test_rrf_handles_an_empty_list():
    assert rrf([[], ["a"]])[0][0] == "a"


def test_gate_rejects_when_both_signals_are_weak():
    assert passes_evidence_gate([hit("n_01", bm25=0.4, cosine=0.1)], 2.0, 0.35) is False


def test_gate_accepts_on_lexical_evidence_alone():
    assert passes_evidence_gate([hit("n_01", bm25=5.0, cosine=0.05)], 2.0, 0.35) is True


def test_gate_accepts_on_semantic_evidence_alone():
    assert passes_evidence_gate([hit("n_01", bm25=0.0, cosine=0.9)], 2.0, 0.35) is True


def test_gate_rejects_an_empty_result_set():
    assert passes_evidence_gate([], 2.0, 0.35) is False


def test_local_search_abstains_with_an_explicit_instruction(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Bread", "sourdough starter hydration")
    result = local_search(conn, embedder, "quantum chromodynamics", config)
    assert result["hits"] == []
    assert "rather than answering from your own knowledge" in result["note"]


def test_local_search_returns_lexical_matches(conn, config):
    # SQLite FTS5's bm25() uses the unsmoothed idf = ln((N - df + 0.5) /
    # (df + 0.5)); with only 2 documents and 1 matching (df = N/2), idf is
    # exactly 0 and the match scores ~0 regardless of how well it matches -
    # a degenerate corpus, not a degenerate query. A ~10-document corpus
    # keeps idf comfortably non-zero so this test actually exercises the
    # production abstain_bm25_floor (2.0) instead of always clearing an
    # empty gate. Do not shrink this back down to 2 documents.
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Scaling", "the plateau is data exhaustion")
    distractors = [
        ("n_02", "Bread", "sourdough starter hydration"),
        ("n_03", "Weather", "cold front moving through tonight"),
        ("n_04", "Garden", "tomatoes need staking this week"),
        ("n_05", "Commute", "the train was delayed again"),
        ("n_06", "Recipe", "simmer the stock for two hours"),
        ("n_07", "Notes", "remember to renew the passport"),
        ("n_08", "Music", "the new album drops on friday"),
        ("n_09", "Repairs", "the bike chain keeps slipping"),
        ("n_10", "Travel", "book the ferry crossing early"),
    ]
    for doc_id, title, text in distractors:
        index_doc(conn, embedder, doc_id, "note", title, text)

    result = local_search(conn, embedder, "data exhaustion", config)
    assert [h["id"] for h in result["hits"]][0] == "n_01"
    assert result["hits"][0]["bm25"] >= config.thresholds.abstain_bm25_floor


def test_local_search_reports_raw_signals_for_tuning(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Scaling", "the plateau is data exhaustion")
    result = local_search(conn, embedder, "data exhaustion", config)
    assert "max_bm25" in result["signals"]
    assert "max_cosine" in result["signals"]


def test_expand_graph_follows_only_traversable_aggregates(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "e_scaling-laws", "entity", "scaling-laws", "scaling laws")
    conn.executemany(
        "INSERT INTO aggregates (key, source, type, target, weight, mean_strength, "
        "traversable) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("r:a|contradicts|scaling-laws", "a", "contradicts", "scaling-laws", 1, 8.0, 1),
            ("r:b|contradicts|scaling-laws", "b", "contradicts", "scaling-laws", 0, 0.0, 0),
        ],
    )
    conn.commit()
    result = local_search(conn, embedder, "scaling laws", config, expand_graph=True)
    neighbours = {n["slug"] for n in result["neighbours"]}
    assert neighbours == {"a"}
