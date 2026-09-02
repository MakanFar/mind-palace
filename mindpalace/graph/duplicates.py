"""The Phase-1 near-duplicate entity lint (spec §10).

Detection only. Spec §8.5 is explicit that aliases do not merge identities,
so nothing here rewrites the graph: the output is a `vault_issue` a human
resolves by hand, normally by adding one slug to the other's `user.aliases`.
That matters more than it looks. `fold` is a pure recompute from the whole
current source set with no incremental path by design, so a merge applied to
the folded tables would be silently undone by the next `sync`. The only
durable place to record "these are the same thing" is the source set itself.

Ported from docling-graph's `alias_reconciler`, minus its merge half.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from mindpalace.graph.fold import Aggregate, GraphTables

_DIGIT_RUNS = re.compile(r"\d+")

# A base shorter than this is a substring of half the vault ("ai", "ml"),
# so containment against it carries no evidence of sameness.
MIN_BASE_LENGTH = 4


@dataclass(frozen=True)
class NearDuplicate:
    """`base` is the shorter slug -- the likelier canonical name of the two."""

    base: str
    superset: str


def digit_signature(slug: str) -> tuple[str, ...]:
    """Ordered digit runs in a slug ("gpt-4-turbo" -> ("4",))."""
    return tuple(_DIGIT_RUNS.findall(slug))


def _labelled_neighbours(
    aggregates: Mapping[str, Aggregate],
) -> dict[str, set[tuple[str, str]]]:
    """slug -> {(edge type, other endpoint)}, ignoring edge direction.

    Direction is deliberately discarded. `aggregate_key` stores a symmetric
    edge type with its endpoints sorted, so whether a node landed in `source`
    or `target` is alphabetical happenstance; a veto that keyed on role would
    fire or not depending on how the two names happened to sort.
    """
    neighbours: dict[str, set[tuple[str, str]]] = {}
    for aggregate in aggregates.values():
        neighbours.setdefault(aggregate.source, set()).add(
            (aggregate.type, aggregate.target)
        )
        neighbours.setdefault(aggregate.target, set()).add(
            (aggregate.type, aggregate.source)
        )
    return neighbours


def propose_near_duplicates(
    tables: GraphTables,
    declared_aliases: Mapping[str, set[str]] | None = None,
) -> list[NearDuplicate]:
    """Slug pairs that look like the same entity written two ways.

    A pair is proposed when one slug contains the other and every guard
    below passes. Each guard exists to suppress a class of false positive
    that would otherwise train the reader to ignore the lint entirely.
    """
    declared = declared_aliases or {}
    slugs = sorted(tables.entities)
    signatures = {slug: digit_signature(slug) for slug in slugs}
    neighbours = _labelled_neighbours(tables.aggregates)

    found: list[NearDuplicate] = []
    for superset in slugs:
        candidates = [
            base
            for base in slugs
            if base != superset
            and len(base) >= MIN_BASE_LENGTH
            and len(base) < len(superset)
            and base in superset
            # "tier-1" is a substring of "tier-10" and they are different
            # tiers. Matching digit runs is what separates a spelling
            # variant from a numbered sibling.
            and signatures[base] == signatures[superset]
        ]
        # Two bases inside one superset is ambiguous ("scaling" and "laws"
        # both sit in "scaling-laws"). Reporting both would be two wrong
        # findings, not one right one.
        if len(candidates) != 1:
            continue
        base = candidates[0]

        # Already reconciled by hand. Re-reporting a resolved pair is how a
        # lint becomes noise.
        if superset in declared.get(base, ()) or base in declared.get(superset, ()):
            continue

        # Sibling co-occurrence veto. Two nodes hanging off one neighbour
        # under one edge type are members of a list something enumerates
        # side by side -- sibling instances, never aliases.
        if neighbours.get(base, set()) & neighbours.get(superset, set()):
            continue

        found.append(NearDuplicate(base=base, superset=superset))

    return sorted(found, key=lambda pair: (pair.base, pair.superset))


@dataclass(frozen=True)
class SimilarEntity:
    """Stage two of resolution (docs/decisions/0001 §6): two profiles whose
    embeddings sit closer than `duplicate_cosine_floor`. `left` sorts first."""

    left: str
    right: str
    similarity: float


def entity_profile(slug: str, descriptions: Iterable[str]) -> str:
    """The text an entity is embedded as for the similarity lint: its name as
    words, then every instance description any note gave it."""
    parts = [slug.replace("-", " ")]
    parts.extend(d.strip() for d in descriptions if d and d.strip())
    return ". ".join(parts)


def propose_similar_entities(
    profiles: Mapping[str, np.ndarray],
    tables: GraphTables,
    floor: float,
    declared_aliases: Mapping[str, set[str]] | None = None,
    kept: Iterable[tuple[str, str]] = (),
) -> list[SimilarEntity]:
    """Entity pairs whose profile embeddings clear `floor`, after the same
    guards as `propose_near_duplicates` plus the `keep` decisions from
    `merges.jsonl`: a pair a human has said is different is never re-proposed.

    Utopia's second stage. Its third -- a model adjudicating the grey zone --
    is the assistant reading the resulting `similar_entity` issue.
    """
    slugs = sorted(slug for slug in profiles if slug in tables.entities)
    if len(slugs) < 2:
        return []
    declared = declared_aliases or {}
    kept_pairs = {tuple(sorted(pair)) for pair in kept}
    neighbours = _labelled_neighbours(tables.aggregates)
    signatures = {slug: digit_signature(slug) for slug in slugs}

    matrix = np.vstack([np.asarray(profiles[slug], dtype=np.float32).ravel() for slug in slugs])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    scores = unit @ unit.T

    found: list[SimilarEntity] = []
    for i, left in enumerate(slugs):
        for j in range(i + 1, len(slugs)):
            right = slugs[j]
            similarity = float(scores[i, j])
            if similarity < floor:
                continue
            if (left, right) in kept_pairs:
                continue
            if right in declared.get(left, ()) or left in declared.get(right, ()):
                continue
            if signatures[left] != signatures[right]:
                continue
            if neighbours.get(left, set()) & neighbours.get(right, set()):
                continue
            found.append(SimilarEntity(left=left, right=right, similarity=similarity))
    return sorted(found, key=lambda pair: (-pair.similarity, pair.left, pair.right))
