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
