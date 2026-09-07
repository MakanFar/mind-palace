# 0003 · A window into the palace

- **Status**: implemented on `feat/obsidian-view`, 2026-09-06; deviations listed at the end
- **Related**: PRD §7.3 (the graph is the one job that needs a visual surface, and it is
  a thin window, not an app), §9 (rent surfaces, own the store); [0001](0001-what-we-borrowed-from-utopia.md)
  for review-gating and the decision log the window writes to; [0002](0002-one-document-for-every-format.md)
  for the text units the panel links to. Utopia's `web/src/pages/Graph.tsx` is the visual
  reference: dashed and faint edges for what is not yet asserted, a side panel with the
  entity's claims and history, a time control over validity dates.

> The vault has entities, proposed edges, claims with dates, and provenance down to the
> text unit, and no way to look at any of it except through an assistant. This record adds
> one visual surface, inside Obsidian, whose first job is to make reviewing feel like
> exploring: every proposal you confirm is a line that appears in the graph you are
> looking at.

## Decisions taken in brainstorming

| Question | Decision |
|---|---|
| First job | Explore and review in one graph: confirmed edges solid, proposed dashed, click a proposal to confirm or dismiss |
| Surface | An Obsidian plugin, since the vault is already an Obsidian vault |
| Data path | The Python side writes `.graph/graph.json` on every sync; the plugin reads it and appends decisions to `.mindpalace/decisions.jsonl` |
| Layout | Full-height graph with a right-hand panel |
| Rendering | d3-force on a 2D canvas, no WebGL; Sigma.js was considered and kept as the visual reference only |
| Freshness | The plugin folds `decisions.jsonl` itself (last write wins per assertion id) and overlays it, so its own decisions show at once without waiting for the Python side to re-fold |
| Code | `obsidian/` folder in this repo, built with esbuild, tested with Vitest |

## Part 1: the contract, `graph.json`

Written by `sync` to `.graph/graph.json` alongside the SQLite cache, so it is Tier 3:
deletable, regenerated on every write, never a source of truth. It carries everything the
window shows and nothing it must compute.

```
{
  "version": 1,
  "generated_at": "2026-09-06T10:00:00Z",
  "entities": [
    {"slug": "scaling-laws", "type": "concept", "rank": 3, "merged_from": [],
     "description": "first line of the entity page, or empty", "stale": true,
     "note_ids": ["n_…"], "text_unit_ids": ["u_…"]}
  ],
  "edges": [
    {"key": "r:a|supports|b", "source": "a", "target": "b",
     "type": "supports", "proposed_type": null, "directed": true,
     "weight": 1, "traversable": true,
     "assertions": [
       {"id": "x_…", "note_id": "n_…", "status": "confirmed", "strength": 7,
        "description": "why", "direction_corrected": false, "text_unit_ids": ["u_…"]}
     ]}
  ],
  "untyped": [
    {"id": "x_…", "source": "a", "target": "b", "proposed_type": "available on",
     "status": "proposed", "note_id": "n_…", "description": "why", "text_unit_ids": []}
  ],
  "claims": [
    {"id": "k_…", "subject": "acme", "text": "…", "status": "confirmed",
     "valid_from": "2015", "valid_to": "2026-03", "supersedes": null, "note_id": "n_…",
     "text_unit_ids": []}
  ],
  "communities": [
    {"lineage_id": "g_…", "level": 0, "parent": null, "members": ["a", "b"],
     "title": "report title or null", "stale": false}
  ],
  "vocabulary": [{"kind": "edge", "proposed": "available-on", "count": 2, "example": "…"}],
  "captures": {"c_…": {"path": "captures/2026-…md", "title": "…"}},
  "notes": {"n_…": {"path": "notes/n_…md"}},
  "units": {"u_…": {"capture_id": "c_…", "locator": "Page 3"}}
}
```

Untyped assertions are listed separately because they form no aggregate (0001 §1); the
window draws them dotted and faint between their endpoints. `captures`, `notes`, and
`units` are lookup tables so the panel can link to files and locate units without a
second read. An entity's `description` is the first line of its page, or empty, so the
window never has to parse an entity page.

`entity_page` links are derived: `entities/<slug>.md`. The window follows the same
conventions as `VaultPaths`, and those are stable.

## Part 2: the plugin

One view, one command, one ribbon icon, one settings tab.

- **View.** An `ItemView` registered under `mind-palace-graph`, opened by the command
  "Open Mind Palace" or the ribbon icon. Left: the canvas. Right: the panel. Top: a bar
  with a search box, entity-type toggles (a legend sorted by count, the way Utopia
  orders its legend), a "proposed" toggle, and an "as of" date field.
- **Reading.** `graph.json` and `decisions.jsonl` are read through the vault adapter,
  because Obsidian's file index does not see dot-folders. The plugin polls both every
  two seconds while the view is open and re-renders when either's modification time
  moves. There is no daemon on the Python side, so this is the only freshness signal.
- **Overlay.** `decisions.jsonl` is folded to `{assertion_id: action}`, last write wins.
  An edge is drawn traversable if `graph.json` says so or if any member assertion is
  confirmed in the overlay; an assertion is drawn dismissed if the overlay says so.
  Nothing else is derived in the plugin. Rank, communities, and provenance wait for the
  next Python sync.
- **Writing.** Confirm and Dismiss append one JSON line to `decisions.jsonl` in exactly
  the shape `DecisionLog.append` writes: `op`, `ts`, `assertion`, `action`, `via`,
  `reason`. `op` is `"obsidian_" + a random id`, `via` is `"obsidian"`. The Python side
  reads the file on its next open, sees the drift, and re-folds. The write uses the
  adapter's append so a concurrent Python write cannot be clobbered; the flock is not
  taken, since Obsidian cannot take it, and a single appended line is the smallest safe
  write there is.
- **Settings.** The path of `graph.json` (default `.graph/graph.json`) and of
  `decisions.jsonl` (default `.mindpalace/decisions.jsonl`), relative to the vault root.

## Part 3: what it looks like

- **Nodes.** A filled circle per entity, colour by type from a fixed palette with the
  `unknown` sentinel in grey, radius from rank (rank 0 is small, never hidden). Label on
  hover and for the focused neighbourhood. Merged-away names never appear; the canonical
  node's tooltip lists them.
- **Edges.** Confirmed solid at full opacity, proposed dashed at half opacity, untyped
  dotted at a quarter, dismissed not drawn. Directed types get an arrowhead. Selecting
  an edge or node draws it and its neighbours in a warm highlight and dims the rest,
  the way Utopia keeps a selection legible without erasing where it came from.
- **Focus.** Typing in the search box and choosing an entity centres it and shows its
  two-hop neighbourhood; escape returns to the overview.
- **As of.** A date in the top bar dims every claim in the panel whose validity does not
  contain it, and the panel says how many are hidden. Edges do not carry validity yet,
  so the canvas does not change.
- **Theme.** Colours come from Obsidian's CSS variables where one exists (background,
  text, accent) so light and dark themes both work; the type palette is fixed.

## Part 4: the panel

For an entity: name, type, rank, description excerpt with "open page"; claims grouped
by status with validity shown as `2015 → 2026-03` and superseded ones struck through;
edges grouped by confirmed, proposed, untyped; the text units it came from as
`Page 3 · capture title`, each opening the capture file. For an edge: each member
assertion with its rationale, strength, note link, and, when proposed, Confirm and
Dismiss buttons with an optional reason field. For an untyped assertion: the same, plus
the wording, and a line saying that adopting the wording is done through the
assistant's `adopt_type`.

## Not in the first version

Merging entities and adopting types from the plugin (assistant tools for now; the
plugin shows the counts). Community zoom levels (needs clustering, which needs 150
entities). Drag-to-link and the "combine" affordance from PRD §7.3. Validity on edges.
Editing anything in an entity page.

## Testing

Python: `sync` writes `graph.json`, and a test checks it against the same tables the
SQLite cache holds. Plugin: Vitest unit tests for the `graph.json` reader, the decision
overlay (including a dismissed-then-confirmed sequence), and the `decisions.jsonl` line
writer; one manual pass in Obsidian against the real vault before merge.

## Deviations found while implementing

- **Traversability and weight are derived from statuses, not OR-ed with the export.**
  The design said "graph.json says so OR any member confirmed"; that is monotone and could
  not retract a dismissal made in the window. The plugin now uses the fold's own rule
  (traversable iff any member confirmed, weight = confirmed count), so it needs nothing
  from the export it cannot recompute.
- **A superseded claim is never overwritten by the overlay.** "Superseded" is derived by
  the Python fold from a later confirmed claim; the log still holds the old claim's own
  confirm, which would otherwise resurrect it.
- **The two readers of `decisions.jsonl` are held to one rule.** The plugin requires the
  same keys Python reads, treats the final line the way `splitlines()` does, and escapes
  U+2028, U+2029 and U+0085 in reasons. `tests/fixtures/decisions-conformance.jsonl` is
  read by both test suites.
- **`sync` hashes source files before folding and embedding**, so a line the window
  appends during the slow embed is still seen as drift on the next open. The window
  still takes no lock.
- **Opening a session regenerates a missing `graph.json`**, and `cluster` re-syncs, so
  "run the server once" is true as written.
- **The layout only re-heats when the node or edge set changes**, the viewport recentres
  only when the focus changes, and the panel re-renders only when the selected item's
  data changed, so a poll during review does not disturb the reader.
- **"Open page" checks the file exists first**; entity pages appear only after a rebuild,
  and opening a missing link would have created a blank note the sync then reports as
  malformed. A future export can carry the page path directly.
