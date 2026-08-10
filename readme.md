# Mind Palace — Product Requirements Document

*Version 0.3 (draft) · Owner: Makan · Status: for review*

> **v0.2 change:** Mind Palace is reframed from a standalone app into a **headless integration layer** — a background sync engine + MCP server over a plain-file vault. It rents its capture and recall surfaces from tools people already use (the OS share sheet, Apple Notes, email, messaging, and AI assistants) and owns only the store and the intelligence. The goal is explicitly **not to make another app** but to disappear into existing workflows.
>
> **v0.3 change:** The knowledge graph adopts the **GraphRAG schema** ([Edge et al., 2024](https://arxiv.org/abs/2404.16130); [microsoft/graphrag](https://github.com/microsoft/graphrag)) — entities, relationships, claims, and, critically, **hierarchical communities with pre-generated community reports**. This adds the one capability the previous design structurally could not deliver: answering *global* questions ("what are the themes in my thinking?") rather than only *local* ones ("what did I save about X?"). Mind Palace extends the schema with **typed, review-gated edges**, which GraphRAG lacks and which the contradiction view depends on.

---

## 1. One-line pitch

**Mind Palace is a local-first "second brain" that plugs into the tools you already use — save anything from any app with the share sheet, and it quietly extracts context, builds an explorable knowledge graph, and lets you recall ideas straight from your AI assistant. No new app to open, no new habit to learn: it's an intelligence layer, not a destination.**

---

## 2. Problem & opportunity

People encounter a steady stream of interesting things every day — a new word, a concept, an article, a post, a paper, a conversation — and almost all of it is lost. The failure isn't a lack of storage; it's that capture, organization, and recall are three separate chores, and the friction between them means most saved material is never revisited, never connected, and never used.

The deeper pain is not "I can't find my note." It's the questions that never get answered because the connective tissue is missing:

- *What does this idea connect to?*
- *Does it contradict something I already believe or saved?*
- *Does it become more meaningful when combined with other ideas?*

Today the workflow splits into two unsatisfying camps:

**The savers.** People download papers to Zotero, screenshot posts, bookmark links, dump notes into Apple Notes or Notion. Capture is easy, but the material sits in silos. "Later, when I have time" rarely comes, and when it does, nothing is connected — each item is an island.

**The gardeners.** Power users live in Obsidian or Roam, hand-linking notes into a graph. The payoff is real — half-formed ideas compound into meaningful ones — but the cost is high: every connection is manual labor, capture is text-first and clunky for images/PDFs/video, and the graph decays the moment you stop tending it.

**The opportunity:** collapse capture, organization, and recall into one low-friction loop where an LLM does the bookkeeping — extracting context, proposing connections, maintaining the graph — while the human keeps ownership of the sources and the thinking. This is the [Karpathy "LLM wiki" pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) (compile knowledge once instead of re-deriving it every time), extended with multi-modal capture and a visual, explorable graph as the primary interface.

---

## 3. Target users & jobs-to-be-done

Primary users are **researchers, builders, entrepreneurs, thinkers, and creatives** — people whose work is producing new ideas from accumulated inputs, and for whom forgetting is a direct tax on output.

**Personas & the job they hire Mind Palace for:**

| Persona | Core job | What "good" looks like |
|---|---|---|
| **The researcher** (grad student, scientist) | "When I read a paper, capture the passages relevant to my thesis and surface them when I'm writing the related section." | A claim in a paper links itself to the argument it supports/contradicts in their draft. |
| **The builder / founder** | "Collect fragments — a competitor move, a user quote, a technical trick — and see the pattern they form before a decision." | Ten loose captures cluster into "this is a wedge for our product." |
| **The writer / creative** | "Hoard half-formed ideas, images, and lines, and let them collide into something." | A quote saved in March resurfaces next to a new note and sparks an essay. |
| **The generalist thinker** | "Never lose a good idea again, and understand how my thinking fits together." | The graph becomes a map of how their interests actually connect. |

**Common thread across all four:** high input volume, high value per connection, low tolerance for manual filing. They will not maintain the system by hand — the tool has to earn its keep by doing the organization *for* them.

---

## 4. Competitive landscape & positioning

The space is crowded but split along two axes: **capture friction** vs. **connection intelligence**, and **your data** vs. **their cloud**.

- **mymind** — beautiful frictionless capture, AI auto-tagging, but deliberately *no* links/graph and fully cloud-locked. Great inbox, no thinking layer.
- **Obsidian / Roam / Logseq** — local (Obsidian), powerful manual graph, but connections are hand-made and capture is text-first. High skill floor.
- **Reflect / Mem** — networked notes with AI, cloud-based, backlinks assisted but graph is secondary.
- **Heptabase** — excellent *visual* whiteboard knowledge base for researchers, but spatial-manual and cloud; you arrange, it doesn't auto-connect.
- **Recall / Napkin-style AI KBs** — auto-summarize and auto-connect, but cloud, opaque, and you don't own the store.

**The white space Mind Palace targets:** *automatic connection intelligence on top of frictionless multi-modal capture, over data you own as plain files.* No single incumbent occupies all three. Obsidian owns "your files" but not auto-connection or easy capture; mymind owns capture but not connection or ownership; Recall owns auto-connection but not ownership. Mind Palace's bet is that **local-first ownership + LLM-maintained graph + one-tap capture of anything** is a defensible combination.

**Positioning statement:** *For thinkers who generate ideas faster than they can organize them, Mind Palace is a local-first knowledge tool that automatically connects everything you capture into an explorable graph — so unlike note apps that make you file and link by hand, or AI apps that lock your mind in someone else's cloud, your second brain organizes itself and stays yours.*

---

## 5. Product principles

1. **Don't build another app — be a layer.** Mind Palace owns only the *store* and the *intelligence*. It **rents** its capture and recall surfaces from tools the user already lives in (share sheet, Apple Notes, email, messaging, AI assistants). Success is measured by how *invisible* it is, not by time-in-app.
2. **Capture must be effortless or it won't happen.** The save action is one share/shortcut from *any* app, any medium, with zero new habit. Organization happens *after* and *automatically* — never a precondition to saving.
3. **The machine does the bookkeeping; the human owns the thinking.** The LLM extracts, summarizes, links, and lints. It never fabricates a source or silently rewrites the user's own words.
4. **Local-first and user-owned.** Everything is plain files (markdown + attachments) on the user's disk. Works offline, versionable with git, portable, no lock-in. The store *is* the moat — and it's what makes every future integration cheap (just another reader/writer of the same files).
5. **Meet recall where thinking already happens.** Recall surfaces inside the user's AI assistant (via MCP) and inside their existing notes/editor — not behind a login to a new product.
6. **The graph and recall are co-equal.** Two first-class jobs: *explore* (wander the visual graph, discover unexpected links) and *recall* (ask a question, get the right idea back, in context). The graph is the one job that needs a thin visual surface; everything else is headless.
7. **Connections are proposed, not imposed — but only durable structure is gated.** The review gate applies to what is *durable and expensive to undo*: entities, relationships, and claims. It does **not** apply to *generated prose* — entity descriptions, community reports — which is machine-written, clearly labeled, and freely replaceable. This distinction is what makes the principle implementable: a single re-clustering can invalidate hundreds of community reports, and no one will ever review those. Discarding and rewriting prose costs only an LLM call; a wrongly-confirmed edge costs a corrupted graph.

Note that "replaceable" is not "derivable". Generated prose cannot be deterministically recreated from the source files — no rebuild can reproduce a model's wording. What the system guarantees instead is **attribution and staleness**: every generated artifact records which sources it was written from and a hash of them, so it can always prove whether it still reflects its inputs and ask to be rewritten when it doesn't.
8. **Compounding, not accumulating.** Value grows with maintenance (dedup, contradiction-resolution, supersession), not just with volume. Avoid "knowledge rot."

---

## 6. Core concepts & data model

Mind Palace adopts the **GraphRAG schema** as its knowledge-graph vocabulary, with two Mind Palace extensions (typed edges and review status). Everything durable is stored as human-readable files so the user is never locked in; the GraphRAG tables themselves are a *derived, rebuildable cache*.

### 6.1 The capture layers

**Capture (a.k.a. "spark")** — the atomic unit of input. A saved thing: a highlight, an image, a link, a voice memo, a paper, a stray thought. Immutable raw payload + metadata (source, timestamp, capture context). *Mirrors Karpathy's "raw sources" layer — the LLM reads these but never edits them.*

**Text unit** — a chunk of a capture, and the unit that extraction actually runs over. For a stray thought, one capture is one text unit and this layer is invisible. For a paper or article it is not, so the layer exists from day one rather than being retrofitted. Every entity, relationship, and claim records the `text_unit_ids` it was extracted from — this is what makes citation-level provenance possible.

**Note / Entry** — LLM- or human-authored markdown derived from captures: a summary, an extracted claim, a synthesis. The wiki layer. Editable.

### 6.2 The graph layers (GraphRAG)

**Entity** — a node in the graph: a person, concept, paper, project, term, theme. Carries a `type`, an LLM-aggregated `description` summarizing every mention across the vault, embeddings of both name and description, the `text_unit_ids` it appears in, `community_ids`, and a `rank` (degree centrality — which doubles as the orphan detector for §7.5's lint pass). Entities get their own page aggregating everything touching them.

**Relationship (edge)** — a link between two entities carrying a free-text `description` of *why* they relate, a numeric `weight` (strength, and duplicate-detection count), and `text_unit_ids` for provenance.

> **Mind Palace extension 1 — typed edges.** GraphRAG relationships are *untyped*: there is no type field and no notion of contradiction. Mind Palace adds a **`type`** (`relates-to`, `supports`, `contradicts`, `example-of`, `part-of`, `derived-from`, `mentions`), declared per type as directed or symmetric. Typed edges are what make the graph reasoning-capable rather than a hairball, and §7.3's contradiction view is impossible without them.
>
> **Mind Palace extension 2 — assertions vs. aggregates.** GraphRAG merges duplicate relationships into one weighted edge. A review-gated system needs both halves separately: an **assertion** is one note's claim that two entities relate, immutable and individually reviewable; the **aggregate** is the derived edge between two entities, whose weight comes from its confirmed assertions. An aggregate becomes traversable once any one of its assertions is confirmed, and two notes may assert the same pair independently — so a dismissal never blocks a later claim backed by new evidence.
>
> Whether `contradicts` should carry clustering weight is an **open experiment, not a settled choice**. Letting it bind is defensible — things you disagree about are things you are thinking about together — but it may equally over-bind unrelated claims that merely share a disputed term. Clustering weight is therefore configurable per edge type, and contradiction surfacing is always an edge-type filter, never a clustering output.

**Claim (covariate)** — a verifiable factual statement about an entity: a date, an event, an assertion. Attached to a subject entity with its own provenance. Optional, and the layer that makes "does this contradict something I already saved?" answerable at the level of specific assertions rather than whole notes.

**Community** — a cluster of densely-connected entities, discovered by **Leiden community detection** run hierarchically: root communities partition the whole graph, and each is recursively subdivided until leaves can no longer be partitioned. Every level is a mutually exclusive, collectively exhaustive partition, which is what makes divide-and-conquer summarization possible. Carries `level`, `parent`, `children`, member ids, `size`, and `period` (the hook for §7.4's temporal recall).

**Community report** — an LLM-written report *per community*, generated bottom-up so higher levels recursively incorporate lower-level summaries. Structured as title, executive summary, an impact `rank` (0–10), and 5–10 detailed findings, each grounded with explicit data citations that must resolve to real graph elements. **These reports are the answer to global questions** — "what are the themes in my thinking?" — and they are pre-computed at index time, not assembled per query. They are generated prose: machine-written, never review-gated, and marked stale rather than silently rewritten when their inputs change (see §P7). Because Leiden reassigns community ids on every run, reports follow a stable **lineage** matched by membership overlap, so a cluster that shifts by one node keeps its report instead of orphaning it.

**Index** — a catalog (`index.md`) the system maintains for human and Obsidian orientation.

**Schema** — the config document (`MINDPALACE.md`, analogous to `CLAUDE.md`/`AGENTS.md`) that defines entity types, edge types, extraction and report templates, and the ingest/query/lint workflows. This is the "contract" that turns a generic model into a disciplined librarian.

### 6.3 Re-clustering policy

Leiden runs over the whole graph and reassigns community ids; community reports are then regenerated for anything that changed. This is the expensive part of the pipeline and it **must not run per capture**. Clustering is triggered by a threshold — N new entities or M new edges since the last run — or on demand. Between runs, new captures are extracted, embedded, and locally searchable; they simply aren't yet part of any community.

**On-disk layout (illustrative):**
```
mind-palace/
  MINDPALACE.md        # schema / config / prompt templates
  index.md             # catalog
  captures/            # immutable raw sources (+ /attachments for images, pdfs, audio)
  notes/               # LLM/human wiki pages
  entities/            # one page per person/concept/paper/project
  communities/         # one page per community report (generated prose)
  .graph/              # derived cache: graph tables, embeddings, assignments
  .mindpalace/
    decisions.jsonl    # append-only review decisions (durable)
    log.jsonl          # append-only operation log for crash recovery / undo
```

Three tiers, and every file belongs to exactly one:

| Tier | Contents | Rebuild behaviour |
|---|---|---|
| **Source** | `captures/`, `notes/`, `decisions.jsonl` | Never regenerated. The complete input for rebuilding all structure. |
| **Generated** | community reports, entity descriptions, user overrides | LLM prose or human intent. Never deleted; carries provenance and a staleness flag. |
| **Cache** | `.graph/`, `index.md` | Deterministically derivable. Delete it all and rebuild byte-identically. |

Keeping these distinct is what makes "your data is plain files" true rather than aspirational — and honest about its limits. Structure is provably rebuildable; prose is provably attributed. Conflating the two produces a guarantee that quietly cannot hold.

---

## 7. Core features

### 7.1 Capture — *rent the surfaces, build none of them*

The design commitment: **Mind Palace ships no capture UI of its own.** Instead it exposes a set of **connectors** that ingest from inboxes the user already uses. Every connector normalizes whatever it receives into a *capture* in the vault. The bar: *if saving requires opening a Mind Palace app, we've failed.*

Launch connectors (in rough priority order):

- **OS Share Sheet + Shortcuts (iOS/macOS)** — the universal one. From *any* app — Safari, Photos, a PDF viewer, a tweet, a highlighted passage — "Share → Save to Mind Palace" drops the item (URL, image, text, file) into the vault. This single connector covers most captures because nearly every app supports the share sheet.
- **Watched folder** — an iCloud/Dropbox folder (or an Obsidian vault) that the engine monitors; anything dropped in is ingested. Also the bridge for desktop drag-and-drop.
- **Email-in** — a personal address ("forward anything here"); great for articles, newsletters, and forwarding from apps that only offer "share via email."
- **Note-to-self bot** — a Telegram/WhatsApp/iMessage-style channel for thumb-typed thoughts and quick voice memos on the go.
- **Apple Notes (via Shortcut)** — a "Send to Mind Palace" Shortcut/automation that pushes selected notes into the vault. *(One-way and automation-based — see the constraint in §11; Notes has no official API.)*

Content handling by medium is unchanged: web links pull the full article body (not just the URL); images get OCR + a vision description; PDFs/papers get title/authors/abstract extraction with highlights becoming claim-notes; audio/video gets transcription with timestamps preserved.

After capture, an **asynchronous ingest pass** (the LLM) runs headlessly: extract context and key entities, write a short summary note, and **propose** links to existing nodes. Nothing blocks the user; results wait in a review queue that itself can be surfaced anywhere (a daily digest note, a message from the bot, or inside the AI assistant).

### 7.2 Organize / Extract — *the automatic librarian*

This is the layer that separates Mind Palace from a bookmark pile. On ingest, the LLM:

- **Extracts context:** what is this, why might it matter, what's it about (entities, terms, claims).
- **Summarizes** long sources into skimmable notes while keeping the raw source intact and linked.
- **Proposes typed connections** to existing entities/notes, with a one-line rationale each.
- **Flags contradictions** ("this claim conflicts with a note you saved in March").
- **Suggests new entities** when a concept recurs across captures.

Extraction follows GraphRAG's method: a single prompt identifies entities (name, type, description) and then the relationships between clearly-related pairs, with **self-reflection "gleaning"** — the extracted set is fed back to the model, which is asked whether anything was missed, then prompted to catch it. The paper shows this roughly doubles entity recall on larger chunks and is what allows bigger chunks without a quality drop. Entity resolution is by normalized-name matching; GraphRAG's own finding is that the pipeline is resilient to residual duplicates because they cluster together anyway.

Durable structure the LLM proposes — entities, relationships, claims — is **review-gated** (see §8, and the risk in §11). Confirmed items become durable graph structure; dismissed ones are remembered so they aren't re-proposed. Derived summaries (entity descriptions, community reports) are not gated; see §P7.

### 7.3 The graph — *explore and think visually*

The one job that genuinely needs a visual surface (text-based capture/recall channels can't draw a graph). Rather than build a heavy app around it, the graph renders in a **thin, occasional "window into the palace"** — either by piggybacking on **Obsidian's graph view** (the vault is already Obsidian-compatible markdown) or as a **lightweight local web view** opened on demand. It's a place you *visit* to think, not a daily driver.

A first-class, interactive canvas — not a decorative "graph view."

**Clustering is not a feature of this view.** Communities are computed at index time and have LLM-written reports attached (§6.2); the graph view *renders* that structure, it does not produce it. This is a change from v0.2, and it matters: it means themes are available to every recall surface — the assistant, the bot, a digest — not only to whoever opens a graph.

- **Visual, navigable graph** of entities and notes, edges colored/styled by type (support vs. contradict vs. relates-to read differently at a glance).
- **Zoom levels map onto the community hierarchy:** constellation (root communities) → neighborhood (a sub-community or one entity + its links) → node (the page itself). Each zoom level has a pre-written report behind it.
- **Direct manipulation:** drag to link, confirm/dismiss proposed edges inline, pin nodes, create a working "board" from a subgraph.
- **Contradiction & tension view:** filter to just `contradicts` edges to find where your own inputs disagree — often the most generative view.
- **"Combine" affordance:** select 2–3 nodes and ask *"what do these form together?"* → LLM drafts a synthesis note, itself added to the graph.

### 7.4 Recall — *the right idea, inside the tool you're already in*

Equally central — and, like capture, surfaced through channels the user already uses rather than a Mind Palace app. The primary recall surface is an **MCP server**, so the vault becomes a queryable memory inside **Claude, ChatGPT (via connectors), Cursor, and any future MCP-speaking assistant**. Secondary surfaces: the note-to-self bot (ask and get an answer back) and inline in the editor/Obsidian.

Recall comes in **three distinct modes**, because local and global questions are structurally different problems. This is the core insight of the GraphRAG paper: vector search retrieves records individually relevant to a query, which by construction cannot answer "what are the main themes here?" — that is query-focused summarization, not retrieval.

- **Local search** — *"what have I saved about diffusion models and memory?"* Hybrid semantic + keyword retrieval seeded on entities, expanded through the graph to their relationships, claims, and source text units. Works whether you remember the exact word or just the gist.
- **Global search** — *"what are the themes in my thinking?" / "where do my inputs disagree?" / "what have I been circling for months without noticing?"* Answered by map-reduce over community reports: each report independently produces a partial answer with a 0–100 self-scored helpfulness rating, zero-scored answers are discarded, and the rest are combined into a final response in descending order of helpfulness. **This is the mode the previous design could not deliver at all**, and the paper measures it beating conventional vector RAG by 72–83% on comprehensiveness and 62–82% on diversity (p<.001), at up to 97% fewer context tokens than summarizing source text directly.
- **Drift search** — a hybrid: start from community reports for orientation, then follow the information scent down into specific entities and captures. The natural mode for "I half-remember something about X, help me find my way back to it."

Every generated answer carries **inline data citations** back to the specific entities, relationships, claims, and reports it drew on — with an explicit rule that nothing may be stated without supporting evidence in the retrieved context. This is the mechanical form of the anti-fabrication commitment in §11, and it means every claim in an answer is clickable back to something the user actually saved.

Beyond the three modes:

- **Contextual resurfacing** — when writing/reading, Mind Palace proactively surfaces related past captures ("you saved 3 things relevant to this"), delivered through whatever surface is active (editor, assistant, digest).
- **Temporal recall** — "what was I thinking about in March?" / "what did I believe about X before I read Y?" using the `period` field on communities and reports plus supersession history.
- **Spaced resurfacing (optional)** — gently re-present neglected-but-valuable ideas so half-formed thoughts get a second life.

### 7.5 Maintain / Lint — *keep the palace from rotting*

A periodic (and on-demand) health pass: detect duplicate captures, contradictory notes, orphaned nodes (no links), stale claims, and gaps (an entity referenced but never fleshed out). Presents a to-do list of fixes the user approves. This is what makes the store *compound* rather than bloat.

Two of these come nearly free from the schema: entity `rank` is degree centrality, so **orphans are rank-0 entities** and no separate detection pass is needed; and graph pruning (dropping low-degree, low-weight noise before clustering) is a standard GraphRAG operation that keeps communities legible as the vault grows.

---

## 8. Key user flows

**Flow A — Capture (the 5-second loop).**
See something interesting → **Share → Save to Mind Palace** from whatever app you're in (or drop it in the watched folder / forward the email / message the bot) → optionally add one line of "why I'm saving this" → done. You never left the app you were in; ingest runs in the background. *Success = user never opened a Mind Palace app and never decided where it goes.*

**Flow B — Review (the daily 5 minutes, delivered to you).**
The review queue comes *to* the user — a daily digest note in the vault, a message from the bot, or a prompt in the AI assistant → for each new capture, see the auto-summary + proposed links/entities → confirm ✓, edit ✎, or dismiss ✗ → confirmed structure enters the graph. *This is the trust-building loop; it replaces hours of manual filing with minutes of judgment, and it has no dedicated app either.*

**Flow C — Explore.**
Open the graph → zoom into a cluster or an entity → follow edges, read node pages inline → spot a surprising `contradicts` or a dense cluster → select nodes → "combine into a note." *Success = user discovers a connection they didn't consciously make.*

**Flow D — Recall in the moment.**
While drafting or thinking, from inside your AI assistant (Claude/ChatGPT/Cursor via MCP) or your editor → ask a question, or get proactive "related to what you're writing" suggestions → pull the exact passage/claim in, with a link back to its source. *Success = the thesis-relevant paragraph from a paper read weeks ago shows up exactly when the relevant section is being written — without opening a separate tool.*

---

## 9. Form factor & architecture — *a headless layer, not an app*

The core architectural decision (v0.2): split the product into three parts and **only own two of them.** Mind Palace owns the **store** and the **intelligence**; it borrows the **capture** and **recall** surfaces from tools the user already uses.

```
   CAPTURE (rented)                STORE + INTELLIGENCE (owned)            RECALL (rented)
 ┌────────────────────┐          ┌──────────────────────────────┐       ┌────────────────────┐
 │ Share sheet /      │          │  Sync engine (watches         │       │ AI assistant       │
 │ Shortcuts          │──────────▶  connectors, normalizes)      │       │ (Claude/ChatGPT/   │
 │ Watched folder     │  writes  │        │                      │◀──────│  Cursor) via MCP   │
 │ Email-in           │  files   │        ▼                      │  MCP  │ Note-to-self bot   │
 │ Note-to-self bot   │          │  Plain-file vault (md + attach)│ query │ Editor / Obsidian  │
 │ Apple Notes (S'cut)│          │  + LLM extract / cluster / lint│       └────────────────────┘
 └────────────────────┘          │  + entities · relationships    │
                                 │    claims · communities        │
                                 │  + community reports (derived) │
                                 │  + local · global · drift      │
                                 └──────────────┬─────────────────┘
                                                │ renders on demand
                                                ▼
                                    Graph view (Obsidian graph OR
                                    lightweight local web view)
```

**The two owned components:**

- **Sync engine (background service).** Watches every connector, normalizes incoming items into captures, and runs the LLM ingest pass. Runs locally (a menubar/background daemon) or optionally as the user's own small hosted service. No UI beyond setup.
- **MCP server.** The single recall/query interface, so any MCP-speaking assistant becomes a front end for the palace. This is what makes "future integrations" nearly free — a new AI surface needs no new Mind Palace code.

Local-first means the *store of truth* is always plain files on the user's disk; compute (the LLM) can be local or cloud (user's choice / BYO-key), but the files never require the cloud to exist or be read.

**Pipeline:**
1. **Capture (connector)** → raw payload written to `captures/` (+ attachments). For images/PDF/audio: OCR / vision / transcription normalizes to text alongside the original.
2. **Chunk** → capture split into text units (a no-op for short thoughts; load-bearing for papers and articles). Text units are the provenance anchor for everything extracted downstream.
3. **Extract (LLM)** → entities, relationship assertions (typed, with description and strength), and claim assertions, using self-reflection gleaning; write summary note; embed. Every assertion enters as `proposed` → queue for review.
   3a. **Fold** → assertions plus the decision log are folded into the derived graph: entities, weighted aggregate edges, ranks. This is a recomputation over the full source set, never an incremental merge — which is what makes re-processing an edited file safe rather than double-counting.
4. **Store** → confirmed entities/relationships/claims written as markdown + front-matter; graph tables and embeddings cached in `.graph/` (fully rebuildable from files).
5. **Cluster (threshold-triggered, not per capture)** → Leiden hierarchical community detection over the graph, then bottom-up LLM generation of a **community report** per community. This is the expensive step; see §6.3.
6. **Retrieve** → **local** (hybrid keyword + vector seeded on entities, expanded through the graph), **global** (map-reduce over community reports with helpfulness scoring), or **drift** (global primer, then local follow-up).
7. **Query (LLM, via MCP)** → synthesize an answer with inline data citations, abstaining where evidence is absent, and file the result back — inside whatever assistant the user asked from.
8. **Lint (LLM, scheduled)** → contradictions, dedup, orphans (rank-0 entities), staleness, graph pruning → review list.

**LLM guardrails (critical):** the model may *propose* but not silently *persist* assertions; every write is attributable and reversible via the append-only log; retrieval below a relevance threshold abstains rather than fabricating; the user's own captured words are never rewritten. (See §11.)

**Portability:** because the durable layer is plain files, the same vault can be opened in Obsidian, versioned in git, or synced across machines — Mind Palace is the intelligence layer, not a jail.

---

## 10. MVP scope & phasing

**Guiding cut:** prove the core loop — *effortless capture → automatic connection → satisfying recall + graph* — on one platform, one primary medium, before breadth.

### Phase 0 — MVP (the headless loop, thin but end-to-end)
- **Background sync engine (macOS-first)** over a plain-files vault — no app UI beyond setup.
- **Two capture connectors: OS Share Sheet/Shortcut + watched folder** (these two cover the vast majority of captures). Media: text/highlights, web links (with article extraction), images (OCR), PDFs. (Audio/video and email-in/bot deferred.)
- Ingest: auto-summary + entity extraction + **proposed typed links** with rationale.
- Review delivered as a **daily digest note** in the vault (no dedicated review app yet).
- **MCP server for recall** — natural-language ask + hybrid search with source links, usable from Claude/Cursor day one. *This is being built first, as a standalone component; see `docs/superpowers/specs/` for its design.*
- **Full GraphRAG schema adopted from day one**, even where a layer is not yet exercised — text units, entity types, claims, community and report tables all exist in v0. Adopting the shape early costs almost nothing; retrofitting it later means migrating a vault.
- **Clustering gated behind an activation threshold.** Leiden and community-report generation switch on only once the graph passes a size threshold (see cold-start risk in §11). Below it, v0 ships local search only.
- Graph: **ride Obsidian's graph view** on the same vault (don't build a custom graph UI yet).
- Storage: markdown + attachments + rebuildable graph cache; git-friendly.
- BYO LLM key (local model optional but not required for v1).

**On cost, honestly.** Graph extraction with gleaning plus community-report generation is the expensive part of this pipeline — GraphRAG's own README warns that indexing "can be an expensive operation," and the paper's 1M-token corpus took 281 minutes of GPT-4-turbo. A personal vault is three orders of magnitude smaller, so absolute cost is small; the thing to watch is not corpus size but **re-clustering frequency**, since each run regenerates reports. Two mitigations exist if it bites: threshold-triggered clustering (§6.3), and a non-LLM extraction path (GraphRAG ships an NLP-based graph extractor) for users who want a cheap or fully-local mode.

*MVP success = a user captures ~50 items over two weeks entirely via the share sheet, never opens a Mind Palace app, and finds that (a) recall from their assistant reliably returns the right item and (b) the graph shows useful connections they didn't make by hand.*

### Phase 1 — Trust & reach
- More connectors: **email-in, note-to-self bot, Apple Notes Shortcut.**
- Lint pass (contradictions, dedup, orphans, staleness).
- Richer review surfaces (bot/assistant-delivered confirm-dismiss). Temporal recall & supersession history.

### Phase 2 — Depth & the graph window
- **Lightweight custom graph web view** (the "window into the palace") with contradiction/tension view and clustering — for when Obsidian's graph isn't enough.
- Audio/video transcription capture.
- Proactive contextual resurfacing while writing/reading.
- "Combine into a note" synthesis; working boards from subgraphs.
- Zotero import; broaden MCP so assistants can *write* (not just read) the palace.

### Phase 3 — Compounding & (optional) collaboration
- Spaced resurfacing. Confidence/quality scoring on LLM-authored notes.
- Optional shared/team palaces.

---

## 11. Risks & mitigations

**Hallucination feedback loop (the #1 risk).** The most-cited critique of the LLM-wiki pattern: *"confident fabrication gets filed back as a page, and then it IS a source."* If the LLM invents a claim and stores it, the error compounds silently. **Mitigation:** review gates on durable structure the LLM asserts as fact (§P7); clear visual distinction between *raw capture* (ground truth), *user note*, and *LLM-inferred* content; `text_unit_ids` provenance on every entity, relationship, and claim; **inline data citations on every generated answer, with an explicit rule that unsupported statements are omitted**; abstain-below-threshold on retrieval; append-only log for undo.

Note the second-order version of this risk, which citations specifically address: the danger is not only a fabricated *file* but a fabricated *provenance* — an answer drawn from the model's own weights that the user believes came from their vault. An answer with no citations is the tell.

**Capture friction kills adoption.** If saving isn't sub-5-seconds and truly any-medium, users default back to screenshots and the graph starves. **Mitigation:** obsess over the rented capture surfaces (share sheet, watched folder) before any fancy graph features; measure time-to-save.

**Closed capture sources (esp. Apple Notes).** Some inboxes the user names — Apple Notes above all — have *no official API*, so "integrate into Notes" can only mean a one-way Shortcut/automation or periodic export, not a live two-way sync. Over-promising deep Notes integration will disappoint. **Mitigation:** treat Notes as *one capture channel via a Shortcut*, not a home for the graph; lead with the share sheet and watched folder (which are open and universal) and frame Notes support honestly as "send to Mind Palace," not "Mind Palace lives in your Notes."

**The graph has no headless home.** Capture and recall can be fully surface-less, but a graph must render *somewhere*, which is in tension with "no new app." **Mitigation:** for MVP, render on Obsidian's existing graph view over the same vault (zero new surface); only later add a thin local web-view "window," kept deliberately occasional-use so it never becomes the daily-driver app we're trying to avoid.

**Dependence on rented surfaces.** Building on others' platforms (share sheet, MCP, messaging bots) means their changes can break us, and MCP is still young. **Mitigation:** the vault is fully functional and portable without any single surface; connectors and recall front-ends are swappable adapters over a stable core, so losing one never orphans the data.

**Review fatigue.** If the queue floods with low-quality proposals, users stop reviewing and trust collapses. **Mitigation:** propose conservatively, rank by confidence, batch, and let dismissals train the system to stop re-proposing.

**Graph-as-hairball.** Untyped, over-connected graphs become noise, not insight (the classic Obsidian graph-view complaint). **Mitigation:** typed edges plus **hierarchical communities**, which give every zoom level a mutually-exclusive, collectively-exhaustive partition with a written report attached — so the graph is readable at every scale by construction rather than by filtering tricks. Graph pruning drops low-degree, low-weight noise before clustering.

**Cold start — a risk the source research does not cover.** GraphRAG was evaluated on two ~1M-token corpora yielding 8,564 and 15,754 entities. A personal vault starts empty and may hold a few hundred short notes after weeks of use. At that scale Leiden produces a handful of communities, root-level reports summarize nearly the entire vault, and global search degenerates into "summarize everything" — which plain context does better and cheaper. **There is no published evidence about the minimum corpus size at which community-level sensemaking starts paying off.** *Mitigation:* gate clustering behind an activation threshold rather than running it from note one; ship local search first; treat finding that threshold empirically as an explicit goal of the dogfooding period, not an implementation detail.

**Local-first vs. compute.** Good LLM extraction may need cloud models, in tension with "your data stays yours." **Mitigation:** data-at-rest is always local files; inference is user-configurable (BYO key or local model); be explicit about what leaves the machine.

**"Nice-to-have" retention.** Second-brain tools notoriously get abandoned. **Mitigation:** the tool must deliver value *without* diligent maintenance — auto-connection and proactive recall have to work on a messy, half-tended vault, because that's the real-world state.

---

## 12. Success metrics

- **Activation:** % of new users who capture ≥10 items and complete ≥1 review session in week 1.
- **Capture friction:** median time-to-save (target < 5s); captures per active user per week.
- **Connection value:** % of proposed links confirmed (proxy for proposal quality); # of user-confirmed connections per capture.
- **Recall efficacy:** % of recall queries where the user opens/uses a returned item; self-reported "found what I needed."
- **Global sensemaking quality:** do community reports name themes the user *recognizes as real* in their own thinking? Measured by rating reports on a recognize / surprising-but-true / wrong scale. This is also how the clustering activation threshold gets found empirically (§11 cold start): the vault size at which "surprising-but-true" starts appearing.
- **Serendipity (the north star):** frequency of "discovered a connection I hadn't made" — via a lightweight in-app signal on the graph/combine surfaces.
- **Retention / compounding:** week-4 and week-12 retention; vault growth *with* maintenance (orphan rate trending down, not up).
- **Invisibility (the anti-metric):** captures and recalls should happen with *near-zero* Mind Palace-app opens. A rising ratio of (captures + recalls) ÷ app-opens means the layer is doing its job by disappearing.

---

## 13. Open questions

1. **Wedge medium:** do we launch narrow (e.g. researchers + papers/highlights) to nail one workflow, or broad multi-medium from day one? (Recommendation: narrow the *go-to-market*, keep the *capture* multi-medium since that's the core promise.)
2. **Review UX:** how much can we auto-confirm high-confidence links to reduce review load without eroding trust? Where's the confidence threshold?
3. ~~**Graph legibility at scale:** what's the right default view at 1,000+ nodes so it stays insight, not hairball?~~ **Answered in v0.3:** hierarchical Leiden communities give a clean partition at every zoom level, each with a written report. *Replacement question:* at what vault size does community-level sensemaking begin to pay for itself — i.e. where should the clustering activation threshold sit? No published evidence exists; this is a dogfooding measurement (see §11 cold start).
4. **Local model viability:** can a small local model do "good enough" extraction for a privacy-max mode, or is BYO-cloud-key the practical default?
5. **Obsidian as the graph runway (mostly answered):** v0.2 leans toward riding Obsidian's graph for MVP rather than building a custom graph app. Open sub-question: how long does Obsidian's graph suffice before the typed-edge / contradiction views force a custom "window"?
6. **How much can stay truly headless?** Setup, connector config, and occasional graph viewing may still need *some* surface. What's the minimum-viable UI, and is it a menubar app, a web dashboard, or config-in-the-vault?
7. **MCP maturity risk:** recall is bet on MCP — acceptable given trajectory, but what's the fallback recall surface if a target assistant lags on MCP support?
8. **Business model:** local-first + BYO-key complicates SaaS pricing — one-time license, sync subscription, or hosted-inference/managed-connectors tier?

---

## 14. Appendix — lineage

Mind Palace has two intellectual parents.

**Andrej Karpathy's [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** — *stop re-deriving, start compiling* — supplies the three layers (raw sources / LLM-owned wiki / schema) and the ingest–query–lint operations. Mind Palace extends it in three ways the source pattern leaves open: **(1) multi-modal capture** (images, PDFs, audio, not just text), **(2) a visual, interactive graph as a co-equal interface** rather than a file tree, and **(3) proactive recall** that surfaces ideas at the moment of need.

**GraphRAG** ([Edge et al., 2024](https://arxiv.org/abs/2404.16130)) supplies the knowledge-graph schema and, more importantly, the insight that **local and global questions are different problems**: retrieval finds records individually relevant to a query and therefore structurally cannot answer "what are the main themes here?" GraphRAG's answer — an LLM-built entity graph, partitioned into hierarchical communities by Leiden, each with a pre-generated report, queried by map-reduce — is what lets Mind Palace answer questions *about the shape of someone's thinking* rather than only about its contents.

Mind Palace departs from GraphRAG in three deliberate ways: **typed, review-gated edges** (GraphRAG relationships are untyped, and contradiction is not expressible in the base schema); **plain markdown as the store of truth** with the graph tables as a rebuildable cache (GraphRAG writes parquet); and **incremental, threshold-triggered clustering** suited to a vault that grows one thought at a time rather than a corpus indexed in one batch.

It also folds in the community's hardest-won lessons — review gates against the hallucination feedback loop, typed edges over free links, hybrid retrieval past the index limit, and supersession/temporal history to fight knowledge rot.