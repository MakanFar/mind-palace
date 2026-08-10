import pytest

from mindpalace.cluster import (
    Community,
    jaccard,
    match_lineages,
    partition,
    should_cluster,
)
from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.graph.fold import Aggregate


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0),
            "mentions": EdgeType("mentions", directed=True, cluster_weight=0.0),
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )


def aggregate(source, target, edge_type="contradicts", traversable=True, weight=1):
    return Aggregate(
        key=f"r:{source}|{edge_type}|{target}",
        source=source,
        type=edge_type,
        target=target,
        weight=weight,
        mean_strength=5.0,
        assertion_ids=("x_1",),
        traversable=traversable,
    )


def test_should_cluster_respects_the_activation_threshold():
    assert should_cluster(149, 150) is False
    assert should_cluster(150, 150) is True


def test_jaccard_of_identical_sets_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_of_empty_sets_is_zero():
    assert jaccard(set(), set()) == 0.0


def test_partition_is_deterministic_under_a_fixed_seed(config):
    aggregates = {
        a.key: a
        for a in [
            aggregate("a", "b"),
            aggregate("b", "c"),
            aggregate("x", "y"),
            aggregate("y", "z"),
        ]
    }
    first = partition(aggregates, config, seed=42)
    second = partition(aggregates, config, seed=42)
    assert [sorted(c.members) for c in first] == [sorted(c.members) for c in second]


def test_partition_ignores_untraversable_aggregates(config):
    aggregates = {
        a.key: a
        for a in [aggregate("a", "b"), aggregate("ghost", "phantom", traversable=False)]
    }
    members = {slug for community in partition(aggregates, config) for slug in community.members}
    assert "ghost" not in members
    assert "a" in members


def test_partition_ignores_zero_weight_edge_types(config):
    """A cluster_weight of 0.0 removes a type from clustering entirely."""
    aggregates = {
        a.key: a for a in [aggregate("a", "b", edge_type="mentions")]
    }
    assert partition(aggregates, config) == []


def test_match_lineages_preserves_id_when_membership_overlaps(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b", "c"}), parent=None)]
    fresh = [Community(lineage_id="g_new", level=0, members=frozenset({"a", "b", "d"}), parent=None)]
    [matched] = match_lineages(fresh, previous, 0.5)
    assert matched.lineage_id == "g_old"
    assert matched.members == frozenset({"a", "b", "d"})


def test_match_lineages_mints_a_new_id_when_overlap_is_too_low(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b", "c"}), parent=None)]
    fresh = [Community(lineage_id="g_new", level=0, members=frozenset({"x", "y", "z"}), parent=None)]
    [matched] = match_lineages(fresh, previous, 0.5)
    assert matched.lineage_id == "g_new"


def test_each_previous_lineage_is_claimed_at_most_once(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b"}), parent=None)]
    fresh = [
        Community(lineage_id="g_1", level=0, members=frozenset({"a", "b"}), parent=None),
        Community(lineage_id="g_2", level=0, members=frozenset({"a", "b"}), parent=None),
    ]
    ids = {community.lineage_id for community in match_lineages(fresh, previous, 0.5)}
    assert "g_old" in ids
    assert len(ids) == 2
