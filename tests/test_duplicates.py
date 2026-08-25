"""The Phase-1 near-duplicate lint promised by spec §10 ("Entity
near-duplicates ... automatic detection is a Phase-1 lint problem").

This is detection only. Nothing here merges anything: spec §8.5 is explicit
that aliases do not merge identities, so the lint's whole output is a
suggestion a human resolves by hand via `user.aliases`.
"""

import pytest

from mindpalace.graph.duplicates import NearDuplicate, propose_near_duplicates
from mindpalace.graph.fold import Aggregate, FoldedEntity, GraphTables


def entity(slug: str) -> FoldedEntity:
    return FoldedEntity(slug=slug, type="concept", rank=0, note_ids=("n_1",))


def aggregate(source: str, edge_type: str, target: str) -> Aggregate:
    return Aggregate(
        key=f"r:{source}|{edge_type}|{target}",
        source=source,
        type=edge_type,
        target=target,
        weight=1,
        mean_strength=5.0,
        assertion_ids=("x_1",),
        traversable=True,
    )


def tables(*slugs: str, aggregates: tuple[Aggregate, ...] = ()) -> GraphTables:
    return GraphTables(
        entities={slug: entity(slug) for slug in slugs},
        aggregates={agg.key: agg for agg in aggregates},
    )


def test_flags_a_slug_contained_in_a_longer_one():
    found = propose_near_duplicates(tables("openai", "openai-inc"))

    assert found == [NearDuplicate(base="openai", superset="openai-inc")]


def test_flags_a_plural_whose_singular_is_a_prefix():
    found = propose_near_duplicates(tables("scaling-law", "scaling-laws"))

    assert found == [NearDuplicate(base="scaling-law", superset="scaling-laws")]


def test_ignores_two_slugs_with_no_containment():
    assert propose_near_duplicates(tables("openai", "anthropic")) == []


def test_ignores_a_pair_whose_digit_runs_differ():
    """`tier-1` is a substring of `tier-10`, but they are different tiers.

    Without the digit-signature guard this pair is proposed, which is the
    single most common false positive a pure-containment lint produces.
    """
    assert propose_near_duplicates(tables("tier-1", "tier-10")) == []


def test_flags_a_pair_whose_digit_runs_match():
    found = propose_near_duplicates(tables("gpt-4", "gpt-4-turbo"))

    assert found == [NearDuplicate(base="gpt-4", superset="gpt-4-turbo")]


def test_ignores_a_base_shorter_than_the_minimum():
    """`ai` is a substring of half the vault; too short to mean anything."""
    assert propose_near_duplicates(tables("ai", "ai-safety")) == []


def test_ignores_a_superset_containing_more_than_one_base():
    """Two candidate bases in one superset is ambiguous, not two findings."""
    assert propose_near_duplicates(tables("scaling", "laws", "scaling-laws")) == []


def test_vetoes_a_pair_that_hangs_off_one_source_under_one_edge_type():
    """Enumerated list members, not aliases -- the sibling co-occurrence veto."""
    found = propose_near_duplicates(
        tables(
            "report",
            "section-1",
            "section-1-a",
            aggregates=(
                aggregate("report", "contains", "section-1"),
                aggregate("report", "contains", "section-1-a"),
            ),
        )
    )

    assert found == []


def test_vetoes_a_pair_that_points_at_one_target_under_one_edge_type():
    found = propose_near_duplicates(
        tables(
            "budget",
            "budget-line",
            "total",
            aggregates=(
                aggregate("budget", "rolls-up-to", "total"),
                aggregate("budget-line", "rolls-up-to", "total"),
            ),
        )
    )

    assert found == []


def test_does_not_veto_when_the_shared_neighbour_uses_different_edge_types():
    """The veto is same-label co-occurrence, not bare adjacency.

    Two nodes hanging off one neighbour under *different* labels are not an
    enumerated list, so the pair must survive to be proposed.
    """
    found = propose_near_duplicates(
        tables(
            "report",
            "section-1",
            "section-1-a",
            aggregates=(
                aggregate("report", "contains", "section-1"),
                aggregate("report", "cites", "section-1-a"),
            ),
        )
    )

    assert found == [NearDuplicate(base="section-1", superset="section-1-a")]


def test_ignores_a_pair_already_declared_as_aliases():
    """Resolved by hand once; the lint must not keep re-reporting it."""
    found = propose_near_duplicates(
        tables("openai", "openai-inc"),
        declared_aliases={"openai": {"openai-inc"}},
    )

    assert found == []


def test_orders_findings_deterministically():
    found = propose_near_duplicates(
        tables("openai-inc", "openai", "anthropic-pbc", "anthropic")
    )

    assert found == [
        NearDuplicate(base="anthropic", superset="anthropic-pbc"),
        NearDuplicate(base="openai", superset="openai-inc"),
    ]


def test_vetoes_a_pair_whose_shared_neighbour_edges_are_stored_opposite_ways():
    """`aggregate_key` sorts endpoints for a symmetric edge type, so which
    node landed in `source` is an artefact of alphabetical order, not of
    meaning. A veto that only checked shared *predecessors* would miss this.
    """
    found = propose_near_duplicates(
        tables(
            "report",
            "section-1",
            "section-1-a",
            aggregates=(
                aggregate("report", "contains", "section-1"),
                aggregate("section-1-a", "contains", "report"),
            ),
        )
    )

    assert found == []
