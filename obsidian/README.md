# Mind Palace for Obsidian

A window into the palace: the vault's entity graph, drawn from `.graph/graph.json`,
with proposals you can confirm or dismiss from the side panel. Decisions are appended to
`.mindpalace/decisions.jsonl`, the same file the Mind Palace server writes. Design:
`docs/decisions/0003-a-window-into-the-palace.md`.

## Install into a vault

```bash
mkdir -p <vault>/.obsidian/plugins/mind-palace
ln -sf "$(pwd)/manifest.json" "$(pwd)/main.js" "$(pwd)/styles.css" <vault>/.obsidian/plugins/mind-palace/
```

Then in Obsidian: Settings, Community plugins, enable Mind Palace. Open the view from
the ribbon icon or the command "Open Mind Palace". The vault needs a `.graph/graph.json`,
which the Mind Palace server writes on every sync; run it once against the vault first.

## Reviewing proposals

Set the top bar's edge mode to "proposed only". The canvas shows just the links with a
proposal left to decide and the panel lists them as a queue. Click a row or press `n`, read
the card, then press `c` to confirm or `d` to dismiss (`r` focuses the reason field first,
Escape leaves it). The view moves on to the next item; a notice offers Undo for a few
seconds, which appends a `reopen` decision. The "order" control groups the queue by note or
sorts it by strength; an edge with several pending assertions has a row that decides them all.

## Retiring an entity

The empty panel lists "Isolated" entities: nothing proposed or confirmed names them. Select
one and press Retire (or `x`) with an optional reason; the node leaves the graph and a
notice offers Undo, which appends a `restore`. Retire is refused while anything live names
the entity: decide a proposal, or dismiss a confirmed edge, first. Dismissing a wrong
relationship also removes the entities it implied, on its own. Design:
`docs/decisions/0004-retiring-an-entity.md`.

## Develop

```bash
npm install   # .npmrc sets legacy-peer-deps: npm's peer resolver fails on the obsidian package
npm run dev        # rebuilds main.js on change
npm run typecheck
```

## Test

```bash
npm test
```
