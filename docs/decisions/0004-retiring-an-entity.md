# 0004 · Retiring an entity

- **Status**: implemented on `feat/obsidian-view`, 2026-09-07; deviations listed at the end
- **Related**: [0001 §5](0001-what-we-borrowed-from-utopia.md) (a merge is a decision, so it
  lives in a log: the model for the retirement log); [0003](0003-a-window-into-the-palace.md)
  (the window, the decision overlay, the review loop this extends).

> Entities are never created directly. The fold derives them from notes: a declared
> instance, or the endpoint of a relationship or claim, whatever that assertion's status.
> So a wrong relationship, once dismissed, leaves two nodes behind, and a mis-extracted
> name declared in one note stays a node forever. There is no way to say "this is not an
> entity". This record adds two things: a fold rule that stops dismissed assertions from
> implying entities, and a logged, reversible decision to retire an entity that has
> nothing live attached to it.

## Decisions taken in brainstorming

| Question | Decision |
|---|---|
| What is an orphan | Not rank 0 (rank counts confirmed edges only, so a fresh vault is all rank 0). *Isolated* means no live assertion or claim touches the slug, proposed included. |
| Implied entities | An endpoint that was never declared in a note and is touched only by dismissed assertions and claims is not folded. Dismissing a wrong relationship removes the nodes it dragged in; reopen brings them back. |
| Retire with live items | Refused. A proposed item gets "decide it first"; a confirmed one gets "dismiss it first". Retiring never drops an assertion, so the review queue stays the only place assertions are decided. |
| Where the decision lives | `.mindpalace/retirements.jsonl`, same shape and rules as `merges.jsonl`: one line per action, last write wins per slug, reversible with `restore`. |
| What outranks a retirement | A live assertion or claim: if a later note proposes a relationship on a retired slug, the entity comes back so the proposal is reviewable, and an integrity finding says so. A bare declaration in a note does not revive it; that is exactly the case retire exists for. |
| Entity pages | Untouched, like a merged-away duplicate's page: rebuild never deletes Tier 2. The page is reported as a `retired_entity_page` issue and not indexed as a live entity. |
| Bulk retire | No. The plugin lists isolated entities and steps through them; each retirement carries its own reason. |

## Part 1: the fold

`fold` takes one more overlay, `retired: Iterable[str]`, beside `merges`. An entity folds
when, after merge resolution:

- it is *declared* (an entity instance in some note) or *live* (an endpoint or subject of
  an assertion or claim whose status is not `dismissed`); and
- it is not retired, or it is live.

Everything else about the fold is unchanged: statuses, aggregates, rank, merged-from,
units. An assertion or claim whose endpoint did not fold is still in the tables (the
window draws nothing for it, since it draws only edges between nodes it has), so nothing
is lost and reopen restores the endpoint on the next fold.

`GraphTables` gains `revived: tuple[str, ...]`: the retired slugs that folded anyway
because something live touched them. `sync` records each as a `vault_issue` of kind
`retired_entity_revived` against `.mindpalace/retirements.jsonl`.

`FoldedEntity` gains `declared: bool`, exported to `graph.json` as `declared`, so the
window can apply the same rule to its own overlay.

## Part 2: the log and the tool

`RetirementLog(path)` in `oplog.py`: `append(slug, action, via, op_id, reason)` with
actions `retire` and `restore`, `retired() -> set[str]`. The line is `{"op", "ts", "slug",
"action", "via", "reason"}` with keys sorted, like every other log. The file is one of the
cache's source files, so an append from the window is drift.

`retire_entity(slug, action="retire", reason=None)`: for `retire`, the slug must be in the
graph and nothing live may touch it; the refusal names what does, with counts. For
`restore`, the slug must currently be retired. Both append under `session.operation` and
rebuild, like `merge_entities`. `graph_stats` gains `isolated` (entities with nothing live
attached) and `retired` (the log's current set) beside `orphans` and `merges`.

## Part 3: the window

- **Overlay.** The plugin folds `retirements.jsonl` like `decisions.jsonl` and applies the
  Part 1 rule to the graph it holds, so a retirement or a dismissal removes the node on the
  plugin's own reload, before the Python side re-folds. It needs `declared` for that.
- **Isolated list.** The empty panel lists "Isolated (N)": entities with nothing live
  attached, any edge mode. A row selects the node.
- **Retire.** The entity panel gets a reason field and a Retire button. When something live
  touches the entity the button is disabled and the panel says why, in the tool's words.
  `x` retires the selected entity from the keyboard under the same rule.
- **Undo.** The notice offers Undo for eight seconds; it appends `restore`. A retired node
  is gone from the canvas, so Undo and the assistant's tool are the two ways back.
- **Settings.** The retirements path, default `.mindpalace/retirements.jsonl`.

## Testing

Python: fold tests for the implied-entity rule (a dismissed-only endpoint vanishes, a
declared one stays, reopen restores), for a retired slug vanishing and a live item reviving
it with the finding; log tests mirroring the merge log; tool tests for the refusals and the
round trip; export test for `declared`; a shared conformance fixture read by both suites.
Plugin: the prune rule, the isolated set, and the refusal rule as pure functions.

## Not in this version

Bulk retirement. Deleting entity pages. Retiring an entity with live items by cascading
dismissals (decided against; see the table).

## Deviations found while implementing

- **Retirements resolve through merges.** `retired` slugs are canonicalised the way every
  other slug is, so retiring a canonical name takes its merged-away aliases with it, and a
  retirement recorded under an alias lands on the canonical entity.
- **The refusal is shown, not just enforced.** The entity panel prints "Cannot retire: …"
  under the disabled button, in the tool's words, so the reader knows what to decide first
  without pressing anything.
- **No existing test relied on dismissed-only endpoints.** The implied-entity rule changed
  no expected behaviour elsewhere in the suite; only new tests describe it.
