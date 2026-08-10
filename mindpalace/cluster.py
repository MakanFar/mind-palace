"""Hierarchical Leiden partitioning with lineage identity across runs.

Leiden reassigns community ids on every run, so identity is tracked by
membership overlap. Without this a single node moving would orphan a report.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from graspologic.partition import hierarchical_leiden

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

    assignments = hierarchical_leiden(graph, random_seed=seed)

    grouped: dict[tuple[int, int], set[str]] = {}
    parents: dict[tuple[int, int], int | None] = {}
    for row in assignments:
        key = (row.level, row.cluster)
        grouped.setdefault(key, set()).add(row.node)
        parents[key] = getattr(row, "parent_cluster", None)

    communities: list[Community] = []
    for (level, cluster), members in sorted(grouped.items()):
        parent = parents[(level, cluster)]
        communities.append(
            Community(
                lineage_id=new_id("g_"),
                level=level,
                members=frozenset(members),
                parent=None if parent is None else f"{level - 1}:{parent}",
            )
        )
    return communities


def match_lineages(
    fresh: list[Community], previous: list[Community], threshold: float
) -> list[Community]:
    unclaimed = list(previous)
    matched: list[Community] = []

    for community in fresh:
        best: Community | None = None
        best_score = 0.0
        for candidate in unclaimed:
            if candidate.level != community.level:
                continue
            score = jaccard(set(community.members), set(candidate.members))
            if score > best_score:
                best, best_score = candidate, score

        if best is not None and best_score >= threshold:
            unclaimed.remove(best)
            matched.append(
                Community(
                    lineage_id=best.lineage_id,
                    level=community.level,
                    members=community.members,
                    parent=community.parent,
                )
            )
        else:
            matched.append(community)

    return matched
