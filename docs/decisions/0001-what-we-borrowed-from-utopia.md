# 0001 · What we borrowed from Utopia

- **Status**: implemented on `feat/utopia-borrowings`, 2026-09-02
- **Source**: [deeplethe/utopia](https://github.com/deeplethe/utopia), an enterprise
  bitemporal knowledge graph in Rust. Its `docs/decisions/` 0003, 0009, 0010,
  0012, 0015 and migrations 0003, 0005, 0018 are the reference for everything below.

> Utopia and Mind Palace disagree about almost everything at the surface
> (Postgres vs. plain files, a server that calls the LLM vs. an MCP client that is
> the LLM) and agree about one thing underneath: the model may propose, but nothing
> it says becomes structure silently. Seven of its mechanisms fit our fold-as-pure-
> recompute model. This record says how each one lands here.

## The constraint every item obeys

`fold` recomputes the graph from the whole source set. Nothing durable may live only
in the SQLite cache. So each borrowed mechanism gets a **source-set home**
(a note field or an append-only log under `.mindpalace/`) and a **cache
projection** the fold derives from it.

| # | Mechanism | Source-set home | Cache projection |
|---|---|---|---|
| 1 | Out-of-vocabulary types are kept, not rejected | `proposed_type` on an entity or relationship assertion; `.mindpalace/vocabulary.jsonl` records adoptions | `assertions.proposed_type`; `vocabulary_proposals` table |
| 2 | Per-item drop ledger | `drops:` list in the note's front-matter | `drops` table |
| 3 | Domain/range on edge types, direction correction | `domain`/`range` on an edge type in MINDPALACE.md; `direction_corrected: true` on the assertion | `assertions.direction_corrected` |
| 4 | Validity time and supersession on claims | `valid_from`, `valid_to`, `supersedes` on a claim assertion | columns on `claims`; status `superseded` |
| 5 | Durable, reversible merges | `.mindpalace/merges.jsonl` (`merge` / `unmerge` / `keep`) | slugs rewritten inside `fold` before anything is touched |
| 6 | Embedding stage in the duplicate lint | `thresholds.duplicate_cosine_floor` (optional, default 0.92) | `similar_entity` vault issues |
| 7 | Echo what landed | — | `write_note` / `propose_relationship` return the landed shape and a `landed` summary line |

## 1. An unknown type is a proposal, not an error (Utopia 0003, 0009, 0010)

`write_note` used to refuse the whole note when one assertion named an edge type
outside `MINDPALACE.md`. The assistant learns to squash "available on" into
`relates-to`, which says nothing. Utopia measured that at 40% of edges before it
changed course.

Now: an entity or relationship whose `type` is not configured is stored with
`type: null` and `proposed_type: "<the model's words>"`. An untyped relationship
forms no aggregate (there is no symmetry or cluster weight to apply) and adds no
rank, but it keeps both endpoints, its rationale, and its review status. An untyped
entity gets the `unknown` sentinel type, as reference-only entities already do.

`sync` counts proposals by normalised wording into `vocabulary_proposals`
(kind, proposed, count, example, ids). `review_queue` lists them under `vocabulary`.

`adopt_type(kind, proposed, name, directed?, cluster_weight?, domain?, range?)`
adds the type to `MINDPALACE.md` if absent and appends
`{action: adopt, kind, proposed, adopted}` to `vocabulary.jsonl`. `fold` reads that
log: an assertion with `type: null` whose `slugify(proposed_type)` is adopted folds
as if it carried the adopted type. Note files are never rewritten. `adopt_type(...,
action="revoke")` appends a revoke; last write wins per proposal.

A note on disk whose `type` names a string absent from config still raises
`UnknownEdgeTypeError` and is quarantined: that is config drift (a type was
removed), not an extraction outcome, and hiding it would hide a real problem.

## 2. Drop the item, not the note (Utopia `extraction_drops`, issue 127)

`write_note` validates each entity, relationship, and claim independently. An item
that fails is dropped with a reason code and the note is still written with the rest.
The note's front-matter carries `drops: [{kind, reason, detail, example}]`, so the
ledger is part of the source set and survives a cache rebuild. `sync` projects it
into a `drops` table; `graph_stats` reports `drops.total` and `drops.by_reason`;
`review_queue` lists recent drops.

Reason codes: `missing_field`, `empty_slug`, `not_an_entity_name` (more than six
words), `bad_strength`, `empty_text`, `self_loop`, `bad_validity`,
`unknown_supersedes`, `domain_mismatch` (item 3). `direction_corrected` is
recorded on the assertion, not as a drop, because nothing was lost.

Note-level failures (empty content, unknown capture) still raise: there is nothing
to write.

## 3. The edge type is a contract (Utopia 0012)

An edge type may declare `domain: [entity types]` and `range: [entity types]`. A
symmetric type may declare `domain` only, applied to both ends. At write time the
endpoint types are resolved from the note's own `entities` first, then from the
graph; an endpoint of unknown type never violates.

If the source violates the domain, the target satisfies it, and the swapped pair is
valid, the endpoints are swapped and the assertion carries
`direction_corrected: true`. If neither orientation is valid the type is dropped
(`type: null`, `proposed_type: <the type>`, drop reason `domain_mismatch`) and the
endpoints and rationale stay. Nothing is silent: both outcomes are echoed (item 7).

`fold` does not re-check signatures. Enforcement happens once, at the write
boundary, which is the one place Utopia lets its ontology enforce anything.

The default template constrains nothing. It documents the keys.

## 4. When a claim held, and what replaced it (Utopia `facts`, migration 0003)

A claim may carry `valid_from` and `valid_to` as `YYYY`, `YYYY-MM`, or
`YYYY-MM-DD`. The string carries its own precision, so there is no separate
precision column; the API reports `valid_from_precision` / `valid_to_precision`
derived from the shape. `valid_to: unknown` means "ended, date unknown", which
Utopia found a single nullable column cannot express. `valid_from` after `valid_to`
is a `bad_validity` drop. A document's date is never written into `valid_to` as an
upper bound.

A claim may name `supersedes: k_...`. When the superseding claim is **confirmed**,
`fold` gives the superseded claim status `superseded` regardless of its own
decision; while it is only proposed the old claim stands. The old claim is never
edited or deleted. `supersedes` must resolve to an existing claim id or the claim is
dropped with `unknown_supersedes`.

`get_entity(name, as_of=None)` filters claims to those whose validity contains
`as_of` when given.

## 5. A merge is a decision, so it lives in a log (Utopia `entity_merges`)

`merge_entities(duplicate, canonical, action="merge"|"unmerge"|"keep", reason)`
appends to `.mindpalace/merges.jsonl`. `fold` resolves every slug through the
current merge map (last write wins per duplicate; chains are followed; a cycle
raises `MergeCycleError`, which the quarantine loop drops with a `merge_cycle`
issue) before touching entities, so the duplicate vanishes from the graph and the
canonical entity inherits its notes, assertions, and claims. `unmerge` restores the
split on the next fold. `keep` records "these are different" so the duplicate lint
stops reporting the pair, the way Utopia caches an adjudication verdict.

Lookup honours merges: `get_entity("old name")` resolves to the canonical entity.

## 6. Names, then embeddings, then the assistant (Utopia migration 0005)

The lint had one stage: slug containment. Stage two embeds an entity profile (slug
plus every instance description) during `sync`, in the same batch as the documents,
and reports pairs whose cosine similarity clears
`thresholds.duplicate_cosine_floor` as `similar_entity` issues, after the same
guards as stage one (declared aliases, `keep` decisions, sibling co-occurrence,
digit signature). Stage three is the assistant: the issue text tells it to
`merge_entities` or to record `keep`.

## 7. Say what landed (Utopia 0015)

Utopia's test: the assistant said "recorded that Acme moved to Shenzhen" while the
graph got an edge with an empty predicate. `write_note` and `propose_relationship`
now return, per item, the type that actually landed (`type`, `proposed_type`,
`direction_corrected`), the drops, and one `landed` sentence such as
"3 entities (1 untyped), 2 relationships (1 direction-corrected, 1 untyped),
1 claim; 2 dropped". The `next` instruction tells the assistant to show that line
to the user rather than restate its intent.

## Not borrowed

Forward-chaining derivation (off by default even in Utopia; a personal vault is too
small to need it), Postgres, the OWL ontology, and the web console.
