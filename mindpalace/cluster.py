"""Hierarchical Leiden partitioning with lineage identity across runs.

Leiden reassigns community ids on every run, so identity is tracked by
membership overlap. Without this a single node moving would orphan a report.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from mindpalace.config import Config
from mindpalace.graph.fold import Aggregate
from mindpalace.ids import new_id


class ClusterBelowThreshold(RuntimeError):
    """Raised when clustering is requested below the activation threshold."""


@dataclass(frozen=True)
class Community:
    lineage_id: str
    level: int
    members: frozenset[str]
    parent: str | None


def should_cluster(entity_count: int, threshold: int) -> bool:
    return entity_count >= threshold


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _build_graph(aggregates: dict[str, Aggregate], config: Config) -> nx.Graph:
    graph = nx.Graph()
    for aggregate in aggregates.values():
        if not aggregate.traversable:
            continue
        edge_type = config.edge_types.get(aggregate.type)
        if edge_type is None or edge_type.cluster_weight == 0.0:
            continue
        weight = max(aggregate.weight, 1) * edge_type.cluster_weight
        if graph.has_edge(aggregate.source, aggregate.target):
            graph[aggregate.source][aggregate.target]["weight"] += weight
        else:
            graph.add_edge(aggregate.source, aggregate.target, weight=weight)
    return graph


def partition(
    aggregates: dict[str, Aggregate], config: Config, seed: int = 42
) -> list[Community]:
    graph = _build_graph(aggregates, config)
    if graph.number_of_edges() == 0:
        return []

    # Imported here, not at module scope: graspologic pulls in umap ->
    # pynndescent -> numba, which cost ~7s of the ~12s `import
    # mindpalace.server` used to take -- inside the fixed window an MCP client
    # allows for `initialize` (30s in Claude Code). Clustering is one tool of
    # fifteen and most sessions never reach this line, so the cost belongs to
    # the caller that needs it.
    from graspologic.partition import hierarchical_leiden

    assignments = hierarchical_leiden(graph, random_seed=seed)

    grouped: dict[tuple[int, int], set[str]] = {}
    parent_cluster_of: dict[tuple[int, int], int | None] = {}
    for row in assignments:
        key = (row.level, row.cluster)
        grouped.setdefault(key, set()).add(row.node)
        parent_cluster_of[key] = getattr(row, "parent_cluster", None)

    ordered_keys = sorted(grouped.items())

    # First pass: mint every community's lineage id before resolving any
    # parent pointers. A parent's (level, cluster) key is not guaranteed to
    # precede its children's in `assignments`, so parent lookups must not
    # depend on the order communities are created in.
    lineage_ids: dict[tuple[int, int], str] = {
        key: new_id("g_") for key, _members in ordered_keys
    }

    communities: list[Community] = []
    for key, members in ordered_keys:
        level, _cluster = key
        parent_cluster = parent_cluster_of[key]
        parent_key = (
            (level - 1, parent_cluster) if parent_cluster is not None else None
        )
        parent_lineage_id = (
            lineage_ids.get(parent_key) if parent_key is not None else None
        )
        communities.append(
            Community(
                lineage_id=lineage_ids[key],
                level=level,
                members=frozenset(members),
                parent=parent_lineage_id,
            )
        )
    return communities


def match_lineages(
    fresh: list[Community], previous: list[Community], threshold: float
) -> list[Community]:
    # Score every (fresh, previous) pair at or above threshold, then assign
    # highest-score-first across the whole set. A greedy pass in `fresh`'s
    # incoming order (Leiden's arbitrary cluster numbering) can let a
    # lower-scoring community claim a previous lineage first, orphaning a
    # later, better-matching (even exact-membership) community that arrives
    # after it -- global assignment avoids that.
    pairs: list[tuple[float, int, int]] = []
    for fi, community in enumerate(fresh):
        for pi, candidate in enumerate(previous):
            if candidate.level != community.level:
                continue
            score = jaccard(set(community.members), set(candidate.members))
            if score >= threshold:
                pairs.append((score, fi, pi))

    # Highest score first; ties broken by original list position (fresh
    # index, then previous index) so the result never depends on dict/set
    # iteration order.
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    assigned_lineage: dict[int, str] = {}
    claimed_previous: set[int] = set()
    for _score, fi, pi in pairs:
        if fi in assigned_lineage or pi in claimed_previous:
            continue
        assigned_lineage[fi] = previous[pi].lineage_id
        claimed_previous.add(pi)

    old_to_new: dict[str, str] = {}
    renamed: list[Community] = []
    for fi, community in enumerate(fresh):
        new_lineage_id = assigned_lineage.get(fi, community.lineage_id)
        old_to_new[community.lineage_id] = new_lineage_id
        renamed.append(
            Community(
                lineage_id=new_lineage_id,
                level=community.level,
                members=community.members,
                parent=community.parent,
            )
        )

    # A matched community's lineage id may have just changed above, which
    # would leave its children's `parent` pointers referring to a now-stale
    # id. Rewrite every parent pointer through the old-to-new map so they
    # keep resolving against the ids this function actually returns.
    return [
        Community(
            lineage_id=community.lineage_id,
            level=community.level,
            members=community.members,
            parent=(
                old_to_new.get(community.parent, community.parent)
                if community.parent is not None
                else None
            ),
        )
        for community in renamed
    ]
