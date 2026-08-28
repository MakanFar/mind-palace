import networkx as nx
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


def test_match_lineages_does_not_orphan_an_exact_membership_continuation(config):
    """A greedy, incoming-order-first pass can let a weaker match claim a
    previous lineage before a later, exact-membership community gets a turn --
    orphaning the report attached to that lineage even though a strictly
    better match existed. `fresh` is deliberately ordered so the weaker match
    (F1, Jaccard 0.75) comes first and the exact match (F2, Jaccard 1.0)
    comes second, mirroring what `partition()` would hand over: fresh
    communities in Leiden's arbitrary cluster-numbering order, unrelated to
    match quality.
    """
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"c", "d", "e"}), parent=None)]
    f1 = Community(lineage_id="g_f1", level=0, members=frozenset({"c", "d", "e", "x"}), parent=None)
    f2 = Community(lineage_id="g_f2", level=0, members=frozenset({"c", "d", "e"}), parent=None)
    fresh = [f1, f2]

    matched = match_lineages(fresh, previous, 0.5)

    exact_match = next(community for community in matched if community.members == f2.members)
    assert exact_match.lineage_id == "g_old"


def _dense_random_aggregates(node_count, edge_probability, seed):
    """A graph dense and large enough that hierarchical_leiden's max_cluster_size
    (1000) forces it to subdivide at least one community, producing real
    level-1 output. Verified empirically: every fixture elsewhere in this file
    is two disconnected 2-edge chains, which only ever yields level 0.
    """
    graph = nx.gnp_random_graph(node_count, edge_probability, seed=seed)
    aggregates = {}
    for u, v in graph.edges():
        key = f"r:n{u}|contradicts|n{v}"
        aggregates[key] = Aggregate(
            key=key,
            source=f"n{u}",
            type="contradicts",
            target=f"n{v}",
            weight=1,
            mean_strength=5.0,
            assertion_ids=("x_1",),
            traversable=True,
        )
    return aggregates


def test_partition_parent_pointers_resolve_through_a_real_hierarchy(config):
    aggregates = _dense_random_aggregates(5000, 0.02, seed=5)
    communities = partition(aggregates, config, seed=42)

    levels = {community.level for community in communities}
    assert max(levels) > 0, "fixture must actually exercise multi-level hierarchical_leiden output"

    lineage_ids = {community.lineage_id for community in communities}
    children = [community for community in communities if community.level > 0]
    assert children
    for child in children:
        assert child.parent in lineage_ids


def test_match_lineages_rewrites_parent_pointers_when_renaming_a_lineage(config):
    aggregates = _dense_random_aggregates(5000, 0.02, seed=5)
    communities = partition(aggregates, config, seed=42)
    assert any(community.level > 0 for community in communities)

    # Give every level-0 (root) community an exact-membership predecessor
    # under a different lineage id, forcing match_lineages to rename every
    # root -- which is exactly the case that leaves stale parent pointers if
    # they are not rewritten.
    previous = [
        Community(lineage_id=f"g_prev_{index}", level=0, members=community.members, parent=None)
        for index, community in enumerate(communities)
        if community.level == 0
    ]

    matched = match_lineages(communities, previous, 0.5)

    roots = [community for community in matched if community.level == 0]
    assert roots and all(community.lineage_id.startswith("g_prev_") for community in roots)

    matched_ids = {community.lineage_id for community in matched}
    children = [community for community in matched if community.level > 0]
    assert children
    for child in children:
        assert child.parent in matched_ids
