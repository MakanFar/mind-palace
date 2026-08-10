# Mind Palace MCP Server — Design Spec

*Date: 2026-08-08 · Owner: Makan · Status: approved for planning*
*Rev 3 — PRD v0.3 (GraphRAG schema) + design review*

Scope: the MCP server only. This is Phase 0 of the PRD (`readme.md` §10), narrowed
further to the one component that can stand alone.

---

## 1. Goal

A standalone MCP server that turns a plain-file vault into a second brain usable
entirely from within an AI assistant. Save a thought by saying "save this"; recall
it weeks later by asking a question. Everything lives in markdown files the user
owns, openable in Obsidian and versionable in git.

The MVP is for **dogfooding by its author** on macOS. Single user, single vault,
no packaging, no config UI, no cross-platform work. Speed to a working loop is the
only priority.

### What this MVP is testing

The PRD's riskiest claim: *automatic typed-edge connection produces links worth
confirming*. Every scope cut below exists to isolate that question. Success is
measured by the confirm-rate on proposed assertions and by whether recall returns
the right note, not by feature count.

A second question rides along, deliberately unanswered at ship time: **at what
vault size does community-level sensemaking start paying off?** No published
evidence exists (PRD §11, cold start). The design's answer is to build the
machinery, keep it inert behind a configurable threshold, and find out.

---

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Purpose | Dogfood for the author | No packaging, config polish, or multi-user concerns |
| Capture media | Text only | Removes scraping, PDF parsing, OCR, transcription entirely |
| Knowledge graph schema | GraphRAG's six tables, adopted whole | Retrofitting the shape later means migrating a vault |
| Edge typing | Mind Palace extension: `type` + review status on assertions | GraphRAG relationships are untyped and cannot express contradiction (PRD §6.2) |
| Durability | Three tiers: Source / Generated / Cache | LLM prose is neither source truth nor deterministically derivable; conflating them made the old rebuild invariant false |
| Assertions vs. aggregates | Modelled separately and explicitly | A per-note assertion and a merged edge have different identity, lifecycle, and review semantics |
| Review decisions | Append-only event log, not in-place mutation | Keeps notes immutable and makes the decision history the audit trail |
| Graph derivation | Pure fold over the full current source set | Incremental merge double-counts on re-detected files |
| Clustering | Built in v0, inert below a configurable threshold | Cold-start risk: on a 50-note vault, root reports summarize everything |
| Retrieval | Local in v0; global switches on with clustering | Per PRD §7.4 |
| Embeddings | Pluggable `Embedder` protocol, local implementation as default | Local-first (§P4) |
| Ingest intelligence | The calling assistant, with a log seam for a later enricher | No LLM inside the server: no key, no cost, deterministic tests |
| Tool surface | Guided primitives (returns carry the next obligation) | Discipline without rigidity; behaviour tunes via strings |
| Runtime | Python | Best embedding and graph-algorithm ecosystem |

---

## 3. Durability model

Three tiers, and every file in the vault belongs to exactly one. This is the
foundation the rest of the design rests on; getting it wrong is what made the
previous revision's central invariant unprovable.

### Tier 1 — Source (never regenerated)

`captures/`, `notes/`, and `.mindpalace/decisions.jsonl`. Immutable after write
by the server; the user may hand-edit them in Obsidian, which is a normal event
(§9). **This tier is the complete input to rebuilding everything else
structural.**

### Tier 2 — Generated (LLM prose and human intent; never deleted)

Community reports, aggregated entity descriptions, and user overrides. These are
model output or human decisions — **not deterministically reproducible**, and
therefore not derivable, no matter how good the rebuild is. They are never deleted
by any rebuild. Each carries:

- `generated_from` — the ids of the source elements it was written from
- `input_hash` — a hash of those elements' content
- `stale` — set true when the current input hash no longer matches

Staleness is the honest replacement for regenerability: the system can always
prove *whether* a piece of prose still reflects its inputs, and can always ask the
assistant to rewrite it, but it cannot recreate it byte-for-byte from files.

### Tier 3 — Cache (deterministically derivable)

`.graph/mindpalace.db`, `index.md`, and the `<!-- mindpalace:related -->` blocks
inside entity pages. Everything here is a pure function of Tier 1. Deleting the
lot and rebuilding must produce byte-identical results — this is the invariant
test in §11, and it is now actually true because prose has been moved out of its
scope.

---

## 4. Data model

The central mapping problem is that GraphRAG's six tables are the parquet output
of a batch pipeline, while Mind Palace needs append-friendly markdown. The
resolution follows GraphRAG's own two-step — extraction produces *element
instances*, which merge into nodes and edges — and makes that split the storage
boundary.

### 4.1 Identifiers

Prefixed ULIDs in front-matter, never derived from filenames, so an Obsidian
rename cannot dangle a reference:

| Prefix | Thing | Notes |
|---|---|---|
| `c_` | capture | |
| `n_` | note | |
| `x_` | relationship assertion | one note's claim that two entities relate |
| `k_` | claim assertion | one note's factual claim about an entity |
| `e_` | entity | `e_<slug>` — the slug *is* the identity |
| `g_` | community lineage | stable across re-clustering runs |
| `op_` | mutation operation | write-ahead log (§10) |

Aggregate relationships have **no ULID**. They are keyed deterministically as
`r:<source_slug>|<type>|<target_slug>`, because they are derived and must be
reproducible by recomputation rather than remembered.

`read(id)` dispatches on prefix. (Communities use `g_` rather than a `c`-prefix
precisely so they cannot be confused with captures.)

### 4.2 Tier 1 — Source

**`captures/YYYY-MM-DD-HHMM-<ulid-suffix>.md`** — immutable raw text. The ULID
suffix is load-bearing: without it, two captures in the same minute collide and
one is lost. One capture is one text unit in v0; the `text_unit` layer exists in
the index from day one so papers and articles don't force a migration in Phase 2.

**`notes/<note-id>-<slug>.md`** — the analysis *and* the record of every element
instance extracted from its capture:

```markdown
---
id: n_01hq
derived_from: c_01hq
created: 2026-08-08T14:22:00Z
author: llm
entities:
  - name: scaling-laws
    type: concept
    description: The empirical relationship between compute, data, and loss.
relationship_assertions:
  - id: x_01ab
    source: scaling-laws
    target: data-exhaustion
    type: contradicts
    strength: 8
    description: This note argues the plateau is a data-supply constraint,
                 contradicting n_01gm's architectural-ceiling account.
claim_assertions:
  - id: k_01cd
    subject: scaling-laws
    text: The perceived plateau reflects data exhaustion, not an
          architectural ceiling.
---

Analysis body — claims, why this matters. Not a restatement of the capture.
```

Assertions carry **no status field**. Status is the fold of the decision log
(§4.3), which is what keeps this file immutable. Writing a capture touches exactly
two files: no write amplification, no git churn across dozens of files per save,
no chance of an ingest write colliding with an Obsidian edit on an entity page.

**`.mindpalace/decisions.jsonl`** — append-only review events. Source tier,
because status is not derivable from notes alone:

```json
{"op": "op_01mn", "ts": "2026-08-08T15:01:00Z", "assertion": "x_01ab",
 "action": "confirm", "via": "resolve_assertion", "reason": "checked both notes"}
```

An assertion's status is the action of its most recent event, or `proposed` if it
has none. Dismissal is not terminal at the *pair* level — see §7.3.

### 4.3 Assertions vs. aggregates

These are different things and the previous revision conflated them.

**`relationship_assertion`** (`x_`) — one note's claim, immutable, source-linked,
individually reviewable. Carries a `strength` (1–10, the assistant's judgement),
which is *not* the aggregate weight.

**`relationship`** — the derived aggregate, keyed `r:<source>|<type>|<target>`,
living only in `.graph/`. Its `weight` is the count of confirmed assertions plus
the mean of their strengths — GraphRAG's "duplicate count becomes edge weight,"
adapted to a review-gated world.

Rules:

- An aggregate is **traversable when at least one of its assertions is
  confirmed.** A ratio or count threshold is premature: at dogfood scale most
  pairs will have exactly one assertion, so a threshold would just be an
  elaborate way of writing "one."
- Confirming or dismissing one assertion never destroys the aggregate — it
  recomputes. Two notes can independently assert the same pair; each is reviewed
  on its own.
- **Symmetric types** (`relates-to`, `contradicts`) normalize by sorting endpoint
  slugs, so `A contradicts B` and `B contradicts A` are one aggregate.
  **Directed types** (`supports`, `example-of`, `part-of`, `derived-from`,
  `mentions`) do not. Directedness is declared per type in `MINDPALACE.md`.

### 4.4 Tier 2 — Generated

**`entities/<slug>.md`** — never deleted by rebuild. Holds LLM-aggregated prose,
user overrides, and one machine-owned block:

```markdown
---
id: e_scaling-laws
type: concept
generated_from: [n_01hq, n_01gm, n_02aa]
input_hash: sha256:9f2a…
stale: false
user:
  type: concept          # overrides the inferred type
  aliases: [scaling law, scaling-law]
---

The empirical relationship between compute, data, and loss…

<!-- mindpalace:related -->
- contradicts [[data-exhaustion]]
- relates-to [[chinchilla]]
<!-- /mindpalace:related -->
```

`rank` and `community_ids` are **not stored here** — they are pure cache and live
in `.graph/` alone, which removes the previous revision's contradiction where a
"derived" file also had to survive deletion to preserve user overrides. Anything
under `user:` wins over inferred values and is never touched.

**`communities/<lineage-id>-<slug>.md`** — one report each: title, executive
summary, impact rank 0–10, findings with citations, plus `cites:`,
`generated_from:`, `input_hash:`, `stale:`, and `lineage_id:`. Only exists above
the activation threshold.

### 4.5 Tier 3 — Cache

**`.graph/mindpalace.db`** — SQLite: FTS5, `sqlite-vec` vectors, and the derived
entity / aggregate-relationship / claim / community tables, plus per-file content
hashes and a `vault_issues` table. **`index.md`** — the human- and
Obsidian-readable catalog.

---

## 5. Architecture

A single long-lived stdio MCP process over one vault directory.

```
Claude (librarian: extracts, writes prose, synthesizes)
      │ MCP stdio
┌─────▼──────────────────────────────────────────────────┐
│ server/      tool defs + guided return payloads         │  ← the "prompt"
├────────────────────────────────────────────────────────┤
│ retrieve/    local (evidence-gated) | global            │
│ cluster/     hierarchical Leiden + lineage matching     │
│ graph/       fold: source records → aggregates          │
├────────────────────────────────────────────────────────┤
│ index/       SQLite (FTS5 + sqlite-vec + derived tables)│
│ embed/       Embedder protocol; local default           │
├────────────────────────────────────────────────────────┤
│ vault/       tiered md I/O, atomic writes, CAS          │
│ schema/      MINDPALACE.md → config + templates         │
│ oplog/       write-ahead ops + decisions                │
└────────────────────────────────────────────────────────┘
      │
   vault/ on disk (git-friendly, Obsidian-openable)
```

Because the assistant is the librarian, **the server's product is its
instructions, not its logic.**

### Modules

**`vault/`** — the only module that touches disk. Owns the three-tier
distinction, layout, id generation, atomic writes (temp + rename), and
compare-and-swap (hash at read, verify before write). Knows nothing about search,
embeddings, graph algorithms, or MCP.

**`schema/`** — parses and validates `MINDPALACE.md` (§6).

**`embed/`** — `Embedder` protocol: `embed(texts) -> ndarray`, `model_id`, `dim`.
`LocalEmbedder` (fastembed) is the default; a cloud implementation sits behind
config. `model_id` and `dim` persist with the vectors; a mismatch triggers a
rebuild.

**`index/`** — owns the SQLite file and all Tier-3 tables.

**`graph/`** — **a pure fold over the full current source set.** Given every
note's assertions plus the decision log, it produces entities, aggregate
relationships, claims, and ranks. It never mutates incrementally, which is what
makes re-detecting a changed file safe: recomputation from scratch cannot
double-count, and deletion needs no special path because the assertions simply
leave the input set. Recomputation is scoped to affected entity slugs for speed,
but the scoped result is required to equal the full-recompute result — an
idempotency property with a test (§11).

**`cluster/`** — hierarchical Leiden via
`graspologic.partition.hierarchical_leiden`, the library the paper used, chosen
for fidelity; it is the heaviest dependency here, and `leidenalg` +
`python-igraph` is the fallback. Seeded for determinism. Runs over **traversable
aggregates only**. Owns the activation threshold and lineage matching (§7.4).

**`retrieve/`** — local and global search. A pure function over `index/` and
`graph/`, so ranking is testable with fixtures.

**`oplog/`** — the write-ahead operation log and the decision log.

**`server/`** — tool definitions and return-payload composition.

### The Phase-1 seam

`save_capture` emits `capture.created`. A future server-side enricher subscribes
to that seam and extracts for captures arriving with no session running, without
`server/` or `vault/` changing shape.

---

## 6. `MINDPALACE.md`

Markdown with YAML front-matter. Front-matter is structured config; the body holds
prompt templates in fenced blocks with named ids.

```markdown
---
schema_version: 1
entity_types: [person, concept, paper, project, term, theme]
edge_types:
  relates-to:   {directed: false, cluster_weight: 1.0}
  contradicts:  {directed: false, cluster_weight: 1.0}
  supports:     {directed: true,  cluster_weight: 1.0}
  example-of:   {directed: true,  cluster_weight: 1.0}
  part-of:      {directed: true,  cluster_weight: 1.0}
  derived-from: {directed: true,  cluster_weight: 1.0}
  mentions:     {directed: true,  cluster_weight: 0.5}
thresholds:
  cluster_activation_entities: 150
  abstain_bm25_floor: 2.0
  abstain_cosine_floor: 0.35
  community_lineage_jaccard: 0.5
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Read any nearest notes you need, then call write_note…
```

Validation is strict: unknown keys are rejected, missing required keys fail
startup with an actionable message naming the key and file, and `schema_version`
gates migrations. Silent defaulting is not allowed — except on fresh scaffold,
where the whole file is written from the built-in default.

`cluster_weight` per edge type exists because **treating `contradicts` as a
positive clustering connection is a hypothesis, not a settled fact.** It may
usefully bind a genuine tension together, or it may over-bind unrelated claims
that merely share a disputed term. Setting a type's weight to `0.0` removes it
from clustering, making this a one-line experiment on real data.

---

## 7. Tool surface

Fifteen tools.

**Write path**

1. `save_capture(text, why?, source?)` — writes the immutable capture; returns
   the guided payload (§7.1)
2. `write_note(derived_from, content, entities[], relationship_assertions[], claim_assertions[])`
   — writes the note with its element instances

**Read path**

3. `local_search(query, k?, expand_graph?)` — §8.2
4. `global_search(query)` — §8.3
5. `read(id)` — full markdown of any capture, note, entity, or community report
6. `neighbors(id, depth?, edge_types?)` — traversal over traversable aggregates,
   grouped by edge type
7. `get_entity(name)` — resolves aliases; the entity page plus everything
   touching it
8. `graph_stats()` — entity / assertion / aggregate / claim counts, orphan count,
   stale-prose count, clustering status and distance to threshold, and the active
   embedder. The tool that answers "is global search worth trying yet?" and "where
   is my data going?"

**Assertion lifecycle**

9. `propose_relationship(source, target, type, description, strength?)` — for
   links spotted outside extraction; creates a standalone assertion
10. `resolve_assertion(id, action: confirm|dismiss, reason?)` — the review gate,
    for both `x_` and `k_` ids, dispatching on prefix. Appends to the decision log
11. `review_queue(limit?)` — two separately-keyed sections: `proposals` (pending
    relationship and claim assertions **ranked by asserted strength**, with both
    endpoints' snippets inline so judging needs no extra `read` calls) and
    `vault_issues` (malformed front-matter, dangling references, alias collisions,
    stale prose). Two kinds of item, two keys — a parse error must never be
    presented as a proposed connection. Ranking is by strength, not retrieval
    score: the queue has no query to score against, so a "retrieval score" here
    would be a number with nothing behind it

**Prose generation**

12. `write_entity_description(slug, description)` — stores the aggregated
    description and clears `stale`
13. `write_community_report(lineage_id, title, summary, rank, findings[], cites[])`
    — validated against §8.4

**Clustering & maintenance**

14. `cluster(force?)` — runs seeded hierarchical Leiden over traversable
    aggregates, matches lineages, and returns communities needing reports with
    their member elements and the report template. Refuses below threshold unless
    `force: true`
15. `rebuild(scope: cache|related_blocks|all)` — regenerates Tier 3. It cannot and
    does not regenerate Tier 2 prose; instead it recomputes input hashes and marks
    prose stale where inputs have moved

### 7.1 Guided payloads

```
save_capture(text: "The plateau talk is mostly about
                    data exhaustion, not architecture.")

→ {
    "id": "c_01hq",
    "path": "captures/2026-08-08-1422-01hq.md",
    "nearest": [
      {"id": "n_01gm", "score": 0.81, "title": "Smooth scaling past 1e26",
       "snippet": "...no evidence of an architectural ceiling; the curve..."},
      {"id": "n_02aa", "score": 0.64, "title": "Chinchilla optimal ratios",
       "snippet": "...compute-optimal token counts imply data becomes..."}
    ],
    "known_entities": [
      {"name": "scaling-laws", "type": "concept", "rank": 7,
       "aliases": ["scaling law"]},
      {"name": "chinchilla", "type": "paper", "rank": 3}
    ],
    "previously_dismissed": [
      {"pair": "scaling-laws|contradicts|chinchilla",
       "reason": "different sense of 'plateau'", "when": "2026-07-30"}
    ],
    "entity_types": ["person", "concept", "paper", "project", "term", "theme"],
    "edge_vocabulary": {"relates-to": "symmetric", "contradicts": "symmetric",
                        "supports": "directed", "...": "..."},
    "next": "<extraction_next template from MINDPALACE.md>"
  }
```

What the payload is doing: the capture is embedded and searched **before the call
returns**, so linking is the cheap path rather than a step to remember;
vocabularies come from `MINDPALACE.md`, so retuning is a file edit;
`known_entities` with ranks and aliases steers reuse over near-duplicate
invention, which is the main defence available given slug-match entity resolution;
`previously_dismissed` **demotes without suppressing** (§7.3); and the template
ends with *"Proposing nothing is a valid outcome"* — without explicit licence to
abstain, a model handed a candidate list links to all of it, which is PRD §11's
review-fatigue risk arriving by the front door.

`cluster()` follows the same pattern, returning member elements plus the report
template and grounding rules rather than a bare community list.

### 7.2 The review gate, honestly

Status is determined by **which tool path was used**, never by a claimed actor:

- `write_note` and `propose_relationship` always produce `proposed`
- `resolve_assertion` is the only route to `confirmed`

There is no actor parameter, because there could be no honest one. **Every MCP
call arrives from the assistant**, so a server cannot verify that a human actually
said "yes, confirm that." This is therefore a **convention backed by audit, not an
authorization boundary**: every decision event records the tool path and the
stated reason, so a bad confirmation is *detectable afterward* rather than
*preventable*. For a single-user dogfood vault that is proportionate; anything
stronger would need an out-of-band confirmation channel, which is not in scope and
would be security theatre if faked. The spec states this plainly rather than
implying a guarantee the architecture cannot provide.

Tier-2 prose is never gated, per PRD §P7.

### 7.3 Dismissal semantics

A dismissal tombstones **that assertion id only** — never the pair or the type.
A later note asserting the same pair is a new assertion, reviewed on its own
merits, because new evidence is exactly the case where a previous "no" should not
bind. `save_capture` surfaces prior dismissals for a pair in
`previously_dismissed`, with the reason, so the assistant can judge whether the
new evidence differs rather than being silently prevented from proposing.

**Dismissal is terminal for the proposal loop, not for the human.** It
permanently stops that assertion from resurfacing as a candidate; it does not
freeze the record. A dismissal can be reversed by an explicit `resolve_assertion`
call, because the alternative — forcing a whole new assertion to undo a mis-click
in a single-user vault — is bad ergonomics for no gain in safety. Status is the
last decision event, and every flip is recorded with its actor path and reason,
so reversal is auditable rather than silent.

### 7.4 Community lineage

Leiden reassigns ids every run, so community identity is tracked by **lineage**,
not membership equality. A new partition's community matches the previous run's by
Jaccard overlap ≥ `community_lineage_jaccard` (default 0.5); on a match it keeps
the lineage id and its report, with `stale: true` if membership changed at all.
Unmatched communities get a fresh lineage. Without this, a single node moving
would orphan a whole report.

---

## 8. Retrieval

### 8.1 What is embedded and indexed

| Artifact | Vector | FTS | Recompute trigger |
|---|---|---|---|
| capture text (= text unit) | ✓ | ✓ | never (immutable) — unless hand-edited |
| note body | ✓ | ✓ | never (immutable) — unless hand-edited |
| entity aggregated description | ✓ | ✓ | on `write_entity_description` |
| community report | ✓ (summary) | ✓ (full content) | on `write_community_report` |
| relationship assertion description | — | ✓ | on note write |

> **Not yet implemented (v0.1).** The recompute-trigger column above describes
> the intended design, not what ships. `sync` currently calls `vectors.clear()`
> and re-embeds every document on every run, so a capture's vector is recomputed
> on each write rather than written once. With the local model and a few hundred
> notes this is seconds per save, growing linearly, in the tool used most. The
> fix is to key vectors by content hash and re-embed only what changed; it was
> deferred out of the final fix wave to avoid touching the sync hot path with a
> single re-review remaining.

Captures and notes being immutable means their vectors are write-once, which
eliminates the most common source of embedding drift. Entity vectors are the only
ones with a live update path, and it is explicit.

### 8.2 Local search

1. Embed the query; vector search over entity descriptions, note bodies, and
   capture texts.
2. FTS over the same set.
3. **Fuse by RRF for ordering only.**
4. Apply the evidence gate.
5. If `expand_graph`, pull one hop from entity hits through traversable
   aggregates.

**The evidence gate does not use the RRF score.** RRF sums `1/(k + rank)`, which
encodes ordering and nothing about relevance — its magnitude shifts with `k`, with
result-list length, and with corpus size, so a fixed threshold over it means
something different every week. Instead, results are returned only if at least one
hit clears `abstain_bm25_floor` on BM25 **or** `abstain_cosine_floor` on raw
cosine similarity — cosine being the only signal here that is comparable across
corpus sizes. Both raw maxima are logged per query so the floors get tuned on real
data rather than guessed once.

Abstention returns:

```json
{"hits": [], "note": "Nothing in the vault is relevant to this query.
                      Say so rather than answering from your own knowledge."}
```

That string matters. Without it the assistant answers from its own weights and the
user believes the answer came from their vault — a fabricated *provenance*, which
is the most insidious form of PRD §11's hallucination loop.

### 8.3 Global search

Reports are ranked by query relevance (vector + FTS) combined with their 0–10
impact rank. **Communities whose report is missing or stale are still returned**,
flagged `report: missing | stale` and carrying their member entity list, so the
assistant can answer from structure instead of a whole region of the graph
silently vanishing from global recall.

On scale, honestly: the paper's map-reduce exists because 1,310 reports cannot fit
in a context window. A personal vault will have ten to twenty root communities,
which fit comfortably — so `global_search` returns ranked reports and the
assistant answers directly. Helpfulness scoring and filtering are specified in the
return payload's instructions but only become load-bearing if reports outgrow
context. Building distributed map-reduce for twelve documents would be theatre.

### 8.4 Citations

Inline, GraphRAG-style: `[Data: Entities (e_scaling-laws); Assertions (x_01ab)]`.
Every generated artifact **also** carries a structural `cites: [ids]` list in
front-matter. `write_community_report` validates every cited id resolves and
rejects the write otherwise; when a later edit orphans a citation, the report is
marked stale rather than left silently wrong. Prose that cannot be checked is
prose that cannot be repaired.

### 8.5 Aliases

`user.aliases` on an entity page feeds a normalized alias → slug map built at
index time. Aliases are indexed for FTS and for entity-name matching during
extraction, and `get_entity` resolves through them. Two entities claiming the same
alias is a `vault_issue`, never silently resolved in favour of one. **Aliases do
not merge identities** — merging two entities is a distinct destructive operation,
deferred past v0.

---

## 9. Reliability

### 9.1 Multi-step atomicity

Atomic rename protects a single file; a save spans markdown, SQLite, the index,
and the log. Each mutating tool call is therefore an operation:

1. Append `op.begin` with an `op_` id and full intent, fsync
2. Write markdown atomically (temp + rename)
3. Apply the SQLite changes in one transaction
4. Append `op.commit`

On startup, any op with a `begin` and no `commit` is replayed. Every operation is
keyed by op id and idempotent, so replay is safe. Because SQLite is Tier 3, the
worst case is always a rebuild rather than data loss.

### 9.2 Concurrency and external edits

A second server on the same vault refuses to start, enforced by an **exclusive
`flock` held on `.mindpalace/lock` for the session's lifetime**. The file also
records the pid, but only as human-readable diagnostics — an `exists()` check
before writing would be check-then-write, and two servers starting together could
both pass it. Mutations are serialized in-process.

Every write is compare-and-swap: the content hash recorded at read time is
verified immediately before the rename. **This detects the common case; it does
not eliminate the race.** An editor that writes between our final hash check and
our rename will still lose its change, and no advisory lock helps because Obsidian
does not take one. CAS is therefore a detector, not a guarantee: it catches edits
made while a tool call is in flight, reports them as conflicts rather than
clobbering silently, and leaves a residual window measured in milliseconds. On a
single-user vault that is a proportionate trade; the honest statement is that the
window exists.

### 9.3 Vault and process contract

`--vault PATH` is required. Scaffolding happens only with an explicit `--init`.
The server refuses a non-empty directory that lacks `MINDPALACE.md`, refuses
`$HOME` and filesystem root outright, and checks `schema_version` at startup.

### 9.4 Network boundary

"Nothing leaves the machine" is true after first run, and the exception should be
stated rather than glossed. On first run the embedding model is downloaded from
the model host — one fetch, hash-pinned — after which the local embedder makes no
network calls. Selecting the cloud embedder **sends every capture and note body to
that provider**; the configuration change is logged, and `graph_stats()` always
reports the active embedder so the boundary is never ambiguous in either
direction.

---

## 10. Failure modes

**Index drift is normal operation.** The vault will be edited in Obsidian — PRD
§7.3 wants that. SQLite stores a content hash per file; startup sweeps mtimes and
re-derives what changed. Because derivation is a fold rather than an incremental
merge, re-processing a file can never double-count its assertions.

> **Partially implemented (v0.1).** Drift is detected and healed at
> `Session.open()` only. The read tools do **not** do a per-call staleness check,
> so an Obsidian edit made while the server is running is not picked up until the
> next restart — search and alias resolution serve the previous content for the
> rest of the session. The drift set does now cover `entities/` and
> `communities/` as well as Tier 1, so the edit is noticed on the next open
> rather than never. Adding a `has_drift`/`resync` call at the top of the read
> tools is the remaining work.

**Deleted or edited notes** — their assertions leave the source set and every
affected aggregate recomputes. Entities left with no assertions become orphans
(rank 0) and are reported, not deleted. Reports and descriptions whose inputs
changed are marked stale.

**Hand-edits to Tier 3 blocks** — the `<!-- mindpalace:related -->` block is
regenerated in place; a header comment says so and points at where durable edits
belong. Tier 1 and Tier 2 hand-edits are respected.

**Malformed front-matter** — the file is still indexed for FTS, recorded in
`vault_issues`, surfaced in `review_queue`. One bad file must never render the
vault unusable.

**Dangling references** — traversal skips them, `review_queue.vault_issues`
reports them. Deferred cleanup, never silent deletion. A *renamed* file does not
dangle: ids live in front-matter.

**Entity near-duplicates** — slug normalization on write, plus `known_entities`
steering reuse. True synonyms are handled by `user.aliases`; automatic detection
is a Phase-1 lint problem.

**First-run model download** — done eagerly at startup with a clear stderr line,
so failure is legible rather than appearing as a mysteriously slow first capture.

**Corrupt SQLite** — detected on open, rebuilt from Tier 1. Nothing is lost.

---

## 11. Testing

The LLM sits outside the server, so every module is deterministic and this is
ordinary software testing rather than eval-wrangling. TDD throughout.

**Unit** — front-matter round-trips including hand-edited-looking input;
`MINDPALACE.md` validation (each rejection path produces a named, actionable
error); assertion→aggregate folding (confirmed count and mean strength feed
weight; symmetric types normalize endpoint order; directed types do not); status
folding from the decision log; RRF ordering against fixture rankings; the evidence
gate; slug and alias normalization; threshold gating.

**Idempotency** — the load-bearing one for §5's fold. Re-deriving from an
unchanged source set produces identical tables. A scoped recompute equals a full
recompute. Re-processing the same note twice does not double weight. Replaying a
committed op changes nothing.

**Clustering** — seeded Leiden is deterministic; lineage matching preserves report
continuity when membership shifts by one node and mints a new lineage when overlap
drops below threshold. Tested on synthetic graphs, since the real vault will not
reach threshold for weeks.

**Crash recovery** — kill between each of §9.1's four steps; assert startup
replay reaches a consistent state every time.

**Concurrency** — a file modified between read-hash and write aborts with a
conflict rather than overwriting.

**Integration** — from an *empty* temp dir so `--init` scaffolding is covered:
`save_capture` → `write_note` → `local_search` finds it → `propose_relationship` →
`review_queue` shows it → `resolve_assertion` confirms it → `neighbors` traverses
it → `cluster(force=true)` partitions it → `write_community_report` →
`global_search` returns it.

**The invariant test — Tier 3 is derivable from Tier 1.** Run an arbitrary
operation sequence, snapshot every query result, delete `.graph/` and all
`mindpalace:related` blocks, `rebuild(all)`, assert identical. This is now
*provable* because prose was moved out of its scope; the companion assertion is
that rebuild leaves every Tier-2 file untouched and marks exactly those whose
input hash moved as stale. If either fails, "your data is plain files" has quietly
become false. These should fail the build hardest.

**Stub embedder** — deterministic hash-based vectors, so the suite never downloads
a model and never depends on embedding quality. One smoke test exercises the real
fastembed path.

**Snapshot tests on guided payload strings** — they are the product, so changing
them should be a deliberate reviewed act.

What tests cannot answer: whether the payloads produce good librarian behaviour,
where the clustering threshold belongs, whether `contradicts` should carry
clustering weight, and where the abstain floors sit. Two weeks of real use answers
all four — which is why each is configuration rather than a constant.

---

## 12. Out of scope for v0

No share sheet or Shortcuts connector, watched folder, email-in, or bot. No
background daemon or sync engine (only the log seam). No web-article extraction,
PDF parsing, OCR, or transcription — though the `text_unit` layer exists so they
don't force a migration. No drift search. No lint pass beyond orphan and staleness
counting. No graph pruning. No entity merging. No automatic synonym detection. No
custom graph view — Obsidian renders the vault. No temporal recall or supersession.
No auto-confirm threshold. No packaging, config UI, multi-vault support, or auth.
No cross-platform support beyond macOS.

---

## 13. Done means

The author can talk to Claude, say "save this," and two weeks later ask "what have
I saved about X" and get the right notes back with citations to their sources —
with everything living in a folder that opens in Obsidian, versions in git, and
reads perfectly well without this server existing.

Clustering will likely still be inert at that point. That is the expected outcome,
not a failure: `graph_stats()` reports the distance to threshold, and whether that
threshold is right becomes the next thing worth measuring.
