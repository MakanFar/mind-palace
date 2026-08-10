import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db
from mindpalace.models import CommunityReport
from mindpalace.retrieve import global_search
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


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
        templates={"report_next": "Ground every finding."},
    )


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


@pytest.fixture
def conn(store):
    connection = db.connect(store.paths.graph_db)
    db.create_schema(connection)
    return connection


def add_community(conn, lineage_id, members, level=0):
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, ?, ?, ?)",
        (lineage_id, level, None, ",".join(sorted(members))),
    )
    conn.commit()


def add_entities(conn, count):
    for index in range(count):
        conn.execute(
            "INSERT INTO entities (slug, type, rank) VALUES (?, 'concept', 1)",
            (f"e{index}",),
        )
    conn.commit()


def test_unavailable_below_threshold_explains_rather_than_errors(conn, store, config):
    add_entities(conn, 3)
    result = global_search(conn, store, StubEmbedder(), "themes?", config)
    assert result["available"] is False
    assert result["entity_count"] == 3
    assert result["threshold"] == 150
    assert "local_search" in result["note"]


def test_available_once_communities_exist(conn, store, config):
    add_entities(conn, 3)
    add_community(conn, "g_01", {"a", "b"})
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="A cluster about scaling limits.",
            rank=7.0,
        )
    )
    result = global_search(conn, store, StubEmbedder(), "scaling", config)
    assert result["available"] is True
    assert result["communities"][0]["lineage_id"] == "g_01"
    assert result["communities"][0]["report"] == "present"


def test_missing_report_is_flagged_and_keeps_its_members(conn, store, config):
    add_community(conn, "g_02", {"x", "y"})
    result = global_search(conn, store, StubEmbedder(), "anything", config)
    [community] = result["communities"]
    assert community["report"] == "missing"
    assert community["members"] == ["x", "y"]


def test_stale_report_is_flagged_but_still_returned(conn, store, config):
    add_community(conn, "g_03", {"a"})
    store.write_report(
        CommunityReport(
            lineage_id="g_03",
            level=0,
            title="Old News",
            summary="Written before the graph moved.",
            rank=4.0,
            stale=True,
        )
    )
    [community] = global_search(conn, store, StubEmbedder(), "news", config)["communities"]
    assert community["report"] == "stale"
    assert community["summary"] == "Written before the graph moved."


def test_higher_impact_rank_breaks_ties(conn, store, config):
    add_community(conn, "g_low", {"a"})
    add_community(conn, "g_high", {"b"})
    for lineage_id, rank in (("g_low", 2.0), ("g_high", 9.0)):
        store.write_report(
            CommunityReport(
                lineage_id=lineage_id,
                level=0,
                title=f"Report {lineage_id}",
                summary="Identical text for both communities.",
                rank=rank,
            )
        )
    result = global_search(conn, store, StubEmbedder(), "identical text", config)
    assert result["communities"][0]["lineage_id"] == "g_high"


def test_instructions_come_from_the_config_template(conn, store, config):
    add_community(conn, "g_04", {"a"})
    result = global_search(conn, store, StubEmbedder(), "q", config)
    assert result["instructions"] == "Ground every finding."
