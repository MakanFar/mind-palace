# Mind Palace MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone stdio MCP server that turns a plain-markdown vault into a GraphRAG-schema knowledge graph, with the calling assistant acting as the extraction and summarization intelligence.

**Architecture:** Three durability tiers (Source: `captures/`+`notes/`+`decisions.jsonl`; Generated: LLM prose with provenance and staleness; Cache: SQLite, deterministically derivable). The graph is a *pure fold* over the full source set — never an incremental merge — so re-processing an edited file cannot double-count. No LLM runs inside the server; every tool returns data plus the next obligation.

**Tech Stack:** Python 3.12+, `uv`, `mcp`, `pyyaml`, `python-ulid`, `numpy`, `fastembed`, `graspologic`, `pytest`.

## Global Constraints

- **Python 3.12+**, managed with `uv`. All commands run via `uv run`.
- **Test framework:** pytest. Every task ends with green tests and a commit.
- **No network in the test suite.** Tests use `StubEmbedder` exclusively. Exactly one test, marked `@pytest.mark.network`, exercises `LocalEmbedder`; it is excluded from the default run via `addopts = "-m 'not network'"`.
- **Determinism is a hard requirement.** Leiden is seeded; embeddings in tests are hash-derived; any test asserting ordering must pin the seed.
- **Vault contract:** `--vault PATH` required; scaffold only under `--init`; refuse a non-empty directory lacking `MINDPALACE.md`; refuse `$HOME` and `/`.
- **IDs are prefixed ULIDs stored in front-matter, never derived from filenames.** Prefixes: `c_` capture, `n_` note, `x_` relationship assertion, `k_` claim assertion, `e_` entity, `g_` community lineage, `op_` operation. Aggregate relationships have **no ULID** — key is `r:<source>|<type>|<target>`.
- **Assertions carry no status field.** Status is the fold of `decisions.jsonl`. Notes are immutable after write.
- **Tier 2 files are never deleted by any rebuild.** `rebuild` recomputes input hashes and marks prose stale; it never regenerates prose.
- **Symmetric edge types** (`relates-to`, `contradicts`) sort endpoint slugs in the aggregate key; **directed types** do not.
- **RRF is used for ordering only.** Abstention is gated on raw BM25 and raw cosine floors, never on the fused score.
- **Every mutating tool call is an operation:** append `op.begin` (fsync) → atomic markdown write → SQLite transaction → append `op.commit`. Replay on startup is idempotent.
- **Tier 2 staleness hashes cover evidence, not identity.** Note bodies, instance and assertion descriptions, weights, claim texts, and statuses all feed the hash. Hashing ids alone would let a rewritten note leave its page reading fresh forever.
- **No helper commits.** `vectors.store`, `vectors.clear`, and `db.write_meta` leave transaction control to the caller; a commit inside a helper splits `sync`'s wipe from its repopulation.
- **The vault lock is `flock`, not an `exists()` check.** Check-then-write lets two servers start together; `flock` is atomic and the kernel releases it when a process dies.
- **Every task ends green.** No task may commit with a known-failing test; if a test needs a later task's code, it belongs in that later task.
- **`fastembed` is a default dependency**, because the shipped `MINDPALACE.md` selects `kind: local` and an optional dependency would mean the default config cannot start the default server.

### Deviation from the spec (deliberate, flagged)

The spec names `sqlite-vec` for vector storage. **This plan stores vectors as a
BLOB column and does brute-force cosine in numpy instead.** Rationale: at MVP
scale (hundreds of notes) brute force is sub-millisecond, and it removes a
loadable-extension dependency that is fragile across macOS Python builds. The
change is confined behind `mindpalace/index/vectors.py`; swapping in `sqlite-vec`
later touches that file only. Everything else in the spec is implemented as
written.

---

## File Structure

```
pyproject.toml
mindpalace/
  __init__.py
  ids.py                 # ULID prefixes, slugify, aggregate keys
  models.py              # record dataclasses + front-matter (de)serialization
  frontmatter.py         # YAML front-matter parse/render
  atomic.py              # atomic write, content hash, compare-and-swap
  config.py              # MINDPALACE.md parse/validate/scaffold
  oplog.py               # write-ahead operation log + decision log
  vault/
    __init__.py
    paths.py             # layout + filename generation
    store.py             # tiered read/write over the vault
  index/
    __init__.py
    db.py                # SQLite DDL + connection
    vectors.py           # vector storage + brute-force cosine
    sync.py              # vault -> cache synchronisation
  graph/
    __init__.py
    fold.py              # (notes, decisions) -> entities/aggregates/claims/ranks
  cluster.py             # hierarchical Leiden + lineage matching
  retrieve.py            # RRF ordering, evidence gate, local + global search
  citations.py           # citation extraction + validation
  rebuild.py             # cache regeneration + staleness marking
  server.py              # MCP tool definitions
  templates/
    MINDPALACE.md        # default scaffold
tests/
  conftest.py
  test_ids.py … test_integration.py
```

Each module owns one responsibility and depends only downward:
`ids` → `models`/`frontmatter` → `atomic` → `vault` → `config`/`oplog` →
`index` → `graph` → `cluster`/`retrieve` → `rebuild` → `server`.

---

## Task 1: Project scaffolding, identifiers, slugs

**Files:**
- Create: `pyproject.toml`, `mindpalace/__init__.py`, `mindpalace/ids.py`
- Test: `tests/test_ids.py`

**Interfaces:**
- Consumes: nothing
- Produces: `new_id(prefix) -> str`, `id_kind(identifier) -> str`, `slugify(name) -> str`, `entity_id(name) -> str`, `aggregate_key(source, edge_type, target, *, symmetric) -> str`, `UnknownIdError`, `PREFIXES: dict[str, str]`

- [ ] **Step 1: Create the project skeleton**

`pyproject.toml`:

```toml
[project]
name = "mindpalace"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "mcp>=1.2.0",
    "pyyaml>=6.0",
    "python-ulid>=2.7",
    "numpy>=1.26",
    "fastembed>=0.4",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[project.scripts]
mindpalace = "mindpalace.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not network'"
markers = ["network: requires network access (model download)"]
```

Then run:

```bash
uv sync
touch mindpalace/__init__.py
mkdir -p tests
```

- [ ] **Step 2: Write the failing test**

`tests/test_ids.py`:

```python
import pytest

from mindpalace.ids import (
    UnknownIdError,
    aggregate_key,
    entity_id,
    id_kind,
    new_id,
    slugify,
)


def test_new_id_carries_prefix_and_is_unique():
    first = new_id("c_")
    second = new_id("c_")
    assert first.startswith("c_")
    assert first != second


def test_new_id_rejects_unknown_prefix():
    with pytest.raises(UnknownIdError):
        new_id("z_")


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("c_01J", "capture"),
        ("n_01J", "note"),
        ("x_01J", "relationship_assertion"),
        ("k_01J", "claim_assertion"),
        ("e_scaling-laws", "entity"),
        ("g_01J", "community"),
        ("op_01J", "operation"),
    ],
)
def test_id_kind_dispatches_on_prefix(identifier, expected):
    assert id_kind(identifier) == expected


def test_id_kind_rejects_unrecognised():
    with pytest.raises(UnknownIdError):
        id_kind("zzz")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Scaling Laws", "scaling-laws"),
        ("  LLM  ", "llm"),
        ("Café Society", "cafe-society"),
        ("data--exhaustion!!", "data-exhaustion"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_entity_id_is_slug_based():
    assert entity_id("Scaling Laws") == "e_scaling-laws"


def test_symmetric_aggregate_key_sorts_endpoints():
    forward = aggregate_key("scaling-laws", "contradicts", "data-exhaustion", symmetric=True)
    reverse = aggregate_key("data-exhaustion", "contradicts", "scaling-laws", symmetric=True)
    assert forward == reverse
    assert forward == "r:data-exhaustion|contradicts|scaling-laws"


def test_directed_aggregate_key_preserves_order():
    forward = aggregate_key("chinchilla", "supports", "scaling-laws", symmetric=False)
    reverse = aggregate_key("scaling-laws", "supports", "chinchilla", symmetric=False)
    assert forward != reverse
    assert forward == "r:chinchilla|supports|scaling-laws"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.ids'`

- [ ] **Step 4: Write the implementation**

`mindpalace/ids.py`:

```python
"""Identifier generation, kind dispatch, and name normalisation."""

from __future__ import annotations

import re
import unicodedata

from ulid import ULID

PREFIXES: dict[str, str] = {
    "c_": "capture",
    "n_": "note",
    "x_": "relationship_assertion",
    "k_": "claim_assertion",
    "e_": "entity",
    "g_": "community",
    "op_": "operation",
}

# Longest prefix first so "op_" is never shadowed by a single-letter prefix.
_ORDERED_PREFIXES = sorted(PREFIXES, key=len, reverse=True)


class UnknownIdError(ValueError):
    """Raised for an unregistered prefix or an unrecognisable identifier."""


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise UnknownIdError(f"unknown id prefix {prefix!r}")
    return f"{prefix}{ULID()}"


def id_kind(identifier: str) -> str:
    for prefix in _ORDERED_PREFIXES:
        if identifier.startswith(prefix):
            return PREFIXES[prefix]
    raise UnknownIdError(f"unrecognised identifier {identifier!r}")


def slugify(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    hyphenated = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower().strip())
    return hyphenated.strip("-")


def entity_id(name: str) -> str:
    return f"e_{slugify(name)}"


def aggregate_key(
    source: str, edge_type: str, target: str, *, symmetric: bool
) -> str:
    left, right = slugify(source), slugify(target)
    if symmetric:
        left, right = sorted((left, right))
    return f"r:{left}|{edge_type}|{right}"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_ids.py -v`
Expected: PASS (12 tests)

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml mindpalace/ tests/
git commit -m "feat: project scaffolding, prefixed ULIDs, slug normalisation"
```

---

## Task 2: Front-matter parsing and record models

**Files:**
- Create: `mindpalace/frontmatter.py`, `mindpalace/models.py`
- Test: `tests/test_frontmatter.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `mindpalace.ids.new_id`
- Produces: `frontmatter.parse(text) -> tuple[dict, str]`, `frontmatter.render(data, body) -> str`, `FrontMatterError`; dataclasses `Capture`, `Note`, `EntityInstance`, `RelationshipAssertion`, `ClaimAssertion`, `EntityPage`, `CommunityReport`, `Decision`; and `note_to_markdown(note) -> str`, `note_from_markdown(text) -> Note`, `capture_to_markdown(capture) -> str`, `capture_from_markdown(text) -> Capture`

- [ ] **Step 1: Write the failing front-matter test**

`tests/test_frontmatter.py`:

```python
import pytest

from mindpalace.frontmatter import FrontMatterError, parse, render


def test_parse_extracts_mapping_and_body():
    text = "---\nid: n_01\nauthor: llm\n---\n\nBody text here.\n"
    data, body = parse(text)
    assert data == {"id": "n_01", "author": "llm"}
    assert body == "Body text here.\n"


def test_parse_returns_empty_mapping_when_absent():
    data, body = parse("Just a body.\n")
    assert data == {}
    assert body == "Just a body.\n"


def test_parse_rejects_unterminated_block():
    with pytest.raises(FrontMatterError, match="unterminated"):
        parse("---\nid: n_01\n\nBody\n")


def test_parse_rejects_non_mapping():
    with pytest.raises(FrontMatterError, match="mapping"):
        parse("---\n- one\n- two\n---\n\nBody\n")


def test_render_round_trips():
    data = {"id": "n_01", "entities": [{"name": "scaling-laws"}]}
    text = render(data, "Body text.")
    parsed, body = parse(text)
    assert parsed == data
    assert body.strip() == "Body text."


def test_render_starts_with_delimiter():
    assert render({"id": "n_01"}, "Body").startswith("---\n")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_frontmatter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.frontmatter'`

- [ ] **Step 3: Implement front-matter**

`mindpalace/frontmatter.py`:

```python
"""YAML front-matter parsing and rendering for vault markdown files."""

from __future__ import annotations

import yaml

DELIMITER = "---"


class FrontMatterError(ValueError):
    """Raised when a file's front-matter block is malformed."""


def parse(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != DELIMITER:
        return {}, text

    for cursor in range(1, len(lines)):
        if lines[cursor].strip() != DELIMITER:
            continue
        raw = "".join(lines[1:cursor])
        body = "".join(lines[cursor + 1 :])
        try:
            loaded = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise FrontMatterError(f"invalid YAML front-matter: {exc}") from exc
        # Default only None to empty. `or {}` here would coerce every falsy
        # non-mapping (False, 0, "", []) into {} *before* the type check, so a
        # malformed document would be silently accepted as having no front-matter.
        data = {} if loaded is None else loaded
        if not isinstance(data, dict):
            raise FrontMatterError("front-matter must be a mapping")
        # Strip exactly the one separator newline `render` writes, not every
        # leading newline — a body may legitimately begin with a blank line.
        return data, body[1:] if body.startswith("\n") else body

    raise FrontMatterError("unterminated front-matter block")


def render(data: dict, body: str) -> str:
    raw = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
    return f"{DELIMITER}\n{raw}{DELIMITER}\n\n{body.rstrip()}\n"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_frontmatter.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing models test**

`tests/test_models.py`:

```python
from mindpalace.models import (
    Capture,
    ClaimAssertion,
    EntityInstance,
    Note,
    RelationshipAssertion,
    capture_from_markdown,
    capture_to_markdown,
    note_from_markdown,
    note_to_markdown,
)

NOTE = Note(
    id="n_01hq",
    derived_from="c_01hq",
    created="2026-08-08T14:22:00Z",
    author="llm",
    body="The plateau is a data-supply constraint.",
    entities=(
        EntityInstance(
            name="scaling-laws",
            type="concept",
            description="Relationship between compute, data, and loss.",
        ),
    ),
    relationship_assertions=(
        RelationshipAssertion(
            id="x_01ab",
            source="scaling-laws",
            target="data-exhaustion",
            type="contradicts",
            strength=8,
            description="Argues plateau is data-driven, not architectural.",
        ),
    ),
    claim_assertions=(
        ClaimAssertion(
            id="k_01cd",
            subject="scaling-laws",
            text="The plateau reflects data exhaustion.",
        ),
    ),
)


def test_note_round_trips_through_markdown():
    assert note_from_markdown(note_to_markdown(NOTE)) == NOTE


def test_note_markdown_omits_status_fields():
    """Status lives in the decision log, never in the immutable note."""
    assert "status" not in note_to_markdown(NOTE)


def test_note_round_trips_with_no_assertions():
    bare = Note(
        id="n_02",
        derived_from="c_02",
        created="2026-08-08T15:00:00Z",
        author="user",
        body="A thought with nothing extracted.",
    )
    assert note_from_markdown(note_to_markdown(bare)) == bare


def test_capture_round_trips_through_markdown():
    capture = Capture(
        id="c_01hq",
        created="2026-08-08T14:22:00Z",
        source="manual",
        why="might matter for the essay",
        text="The plateau talk is mostly about data exhaustion.",
    )
    assert capture_from_markdown(capture_to_markdown(capture)) == capture


def test_capture_round_trips_without_optional_why():
    capture = Capture(
        id="c_02",
        created="2026-08-08T16:00:00Z",
        source="manual",
        why=None,
        text="A bare thought.",
    )
    assert capture_from_markdown(capture_to_markdown(capture)) == capture
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.models'`

- [ ] **Step 7: Implement the models**

`mindpalace/models.py`:

```python
"""Record types for every artifact the vault stores."""

from __future__ import annotations

from dataclasses import dataclass, field

from mindpalace.frontmatter import parse, render


@dataclass(frozen=True)
class EntityInstance:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class RelationshipAssertion:
    id: str
    source: str
    target: str
    type: str
    strength: int
    description: str


@dataclass(frozen=True)
class ClaimAssertion:
    id: str
    subject: str
    text: str


@dataclass(frozen=True)
class Capture:
    id: str
    created: str
    source: str
    why: str | None
    text: str


@dataclass(frozen=True)
class Note:
    id: str
    derived_from: str
    created: str
    author: str
    body: str
    entities: tuple[EntityInstance, ...] = ()
    relationship_assertions: tuple[RelationshipAssertion, ...] = ()
    claim_assertions: tuple[ClaimAssertion, ...] = ()


@dataclass
class EntityPage:
    """Tier 2. Never deleted by rebuild; `related` is the one machine-owned block."""

    slug: str
    type: str
    description: str
    generated_from: list[str] = field(default_factory=list)
    input_hash: str = ""
    stale: bool = True
    user: dict = field(default_factory=dict)
    related: list[str] = field(default_factory=list)


@dataclass
class CommunityReport:
    """Tier 2. Keyed by stable lineage id, not by Leiden's per-run community id."""

    lineage_id: str
    level: int
    title: str
    summary: str
    rank: float
    findings: list[dict] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)
    generated_from: list[str] = field(default_factory=list)
    input_hash: str = ""
    stale: bool = False


@dataclass(frozen=True)
class Decision:
    op: str
    ts: str
    assertion: str
    action: str
    via: str
    reason: str | None = None


def capture_to_markdown(capture: Capture) -> str:
    data = {"id": capture.id, "created": capture.created, "source": capture.source}
    if capture.why is not None:
        data["why"] = capture.why
    return render(data, capture.text)


def capture_from_markdown(text: str) -> Capture:
    data, body = parse(text)
    return Capture(
        id=data["id"],
        created=data["created"],
        source=data["source"],
        why=data.get("why"),
        text=body.rstrip("\n"),
    )


def note_to_markdown(note: Note) -> str:
    data: dict = {
        "id": note.id,
        "derived_from": note.derived_from,
        "created": note.created,
        "author": note.author,
    }
    if note.entities:
        data["entities"] = [
            {"name": e.name, "type": e.type, "description": e.description}
            for e in note.entities
        ]
    if note.relationship_assertions:
        data["relationship_assertions"] = [
            {
                "id": r.id,
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "strength": r.strength,
                "description": r.description,
            }
            for r in note.relationship_assertions
        ]
    if note.claim_assertions:
        data["claim_assertions"] = [
            {"id": c.id, "subject": c.subject, "text": c.text}
            for c in note.claim_assertions
        ]
    return render(data, note.body)


def note_from_markdown(text: str) -> Note:
    data, body = parse(text)
    return Note(
        id=data["id"],
        derived_from=data["derived_from"],
        created=data["created"],
        author=data["author"],
        body=body.rstrip("\n"),
        entities=tuple(
            EntityInstance(name=e["name"], type=e["type"], description=e["description"])
            for e in data.get("entities", [])
        ),
        relationship_assertions=tuple(
            RelationshipAssertion(
                id=r["id"],
                source=r["source"],
                target=r["target"],
                type=r["type"],
                strength=r["strength"],
                description=r["description"],
            )
            for r in data.get("relationship_assertions", [])
        ),
        claim_assertions=tuple(
            ClaimAssertion(id=c["id"], subject=c["subject"], text=c["text"])
            for c in data.get("claim_assertions", [])
        ),
    )
```

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (5 tests)

- [ ] **Step 9: Commit**

```bash
git add mindpalace/frontmatter.py mindpalace/models.py tests/test_frontmatter.py tests/test_models.py
git commit -m "feat: front-matter parsing and vault record models"
```

---

## Task 3: Atomic writes, content hashing, compare-and-swap

**Files:**
- Create: `mindpalace/atomic.py`
- Test: `tests/test_atomic.py`

**Interfaces:**
- Consumes: nothing
- Produces: `content_hash(text) -> str`, `atomic_write(path, content) -> None`, `cas_write(path, content, expected_hash) -> None`, `ConflictError`

- [ ] **Step 1: Write the failing test**

`tests/test_atomic.py`:

```python
import pytest

from mindpalace.atomic import ConflictError, atomic_write, cas_write, content_hash


def test_content_hash_is_stable_and_prefixed():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello").startswith("sha256:")
    assert content_hash("hello") != content_hash("world")


def test_atomic_write_creates_file_and_parents(tmp_path):
    target = tmp_path / "nested" / "note.md"
    atomic_write(target, "body\n")
    assert target.read_text() == "body\n"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "body\n")
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_cas_write_succeeds_when_hash_matches(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "original\n")
    cas_write(target, "updated\n", content_hash("original\n"))
    assert target.read_text() == "updated\n"


def test_cas_write_aborts_when_file_changed_underneath(tmp_path):
    target = tmp_path / "note.md"
    atomic_write(target, "original\n")
    stale = content_hash("original\n")
    atomic_write(target, "edited in obsidian\n")  # external edit

    with pytest.raises(ConflictError, match="changed on disk"):
        cas_write(target, "updated\n", stale)
    assert target.read_text() == "edited in obsidian\n"


def test_cas_write_with_none_expects_absent_file(tmp_path):
    target = tmp_path / "new.md"
    cas_write(target, "fresh\n", None)
    assert target.read_text() == "fresh\n"


def test_cas_write_with_none_rejects_existing_file(tmp_path):
    target = tmp_path / "new.md"
    atomic_write(target, "already here\n")
    with pytest.raises(ConflictError, match="already exists"):
        cas_write(target, "fresh\n", None)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_atomic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.atomic'`

- [ ] **Step 3: Implement it**

`mindpalace/atomic.py`:

```python
"""Crash-safe single-file writes with optimistic concurrency control."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class ConflictError(RuntimeError):
    """Raised when a file changed between read and write."""


def content_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def atomic_write(path: Path, content: str) -> None:
    """Write via temp file + rename so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def cas_write(path: Path, content: str, expected_hash: str | None) -> None:
    """Write only if the on-disk content still hashes to `expected_hash`.

    `expected_hash=None` asserts the file does not yet exist. A mismatch aborts
    rather than clobbering an edit made in Obsidian since we last read.
    """
    if expected_hash is None:
        # Exclusive create at the OS level, not by pre-check. An exists() test
        # followed by a rename is check-then-act: the file can appear in the gap
        # and the rename overwrites it. os.link fails atomically instead, which
        # is what makes "an immutable capture is never overwritten" a guarantee
        # rather than a hope.
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                raise ConflictError(f"{path} already exists") from None
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        else:
            temp_path.unlink()
    else:
        # This branch stays a detector by design (spec §9.2): it catches an edit
        # made while a call is in flight, and cannot close the window entirely.
        if not path.exists():
            raise ConflictError(f"{path} changed on disk: expected content, found none")
        actual = content_hash(path.read_text(encoding="utf-8"))
        if actual != expected_hash:
            raise ConflictError(
                f"{path} changed on disk: expected {expected_hash}, found {actual}"
            )
        atomic_write(path, content)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_atomic.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/atomic.py tests/test_atomic.py
git commit -m "feat: atomic writes with content-hash compare-and-swap"
```

---

## Task 4: Vault layout and tiered store

**Files:**
- Create: `mindpalace/vault/__init__.py`, `mindpalace/vault/paths.py`, `mindpalace/vault/store.py`
- Test: `tests/test_vault_paths.py`, `tests/test_vault_store.py`

**Interfaces:**
- Consumes: `mindpalace.atomic`, `mindpalace.models`, `mindpalace.ids.slugify`
- Produces: `VaultPaths(root)` with attributes `root, mindpalace_md, index_md, captures, notes, entities, communities, graph_db, decisions_log, op_log, lock` and methods `capture_path(capture_id, created) -> Path`, `note_path(note_id, slug) -> Path`, `entity_path(slug) -> Path`, `community_path(lineage_id) -> Path`; `VaultStore(paths)` with `write_capture`, `read_capture`, `write_note`, `read_note`, `iter_captures`, `iter_notes`, `read_entity_page`, `write_entity_page`, `iter_entity_pages`, `read_report`, `write_report`, `iter_reports`

- [ ] **Step 1: Write the failing paths test**

`tests/test_vault_paths.py`:

```python
from datetime import UTC, datetime

from mindpalace.vault.paths import VaultPaths


def test_capture_filename_includes_ulid_suffix(tmp_path):
    paths = VaultPaths(tmp_path)
    created = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    path = paths.capture_path("c_01HQABCDEF", created)
    assert path.parent == tmp_path / "captures"
    assert path.name == "2026-08-08-1422-ABCDEF.md"


def test_two_captures_in_the_same_minute_do_not_collide(tmp_path):
    """Without the ULID suffix one capture would silently overwrite the other."""
    paths = VaultPaths(tmp_path)
    created = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    first = paths.capture_path("c_01HQAAAAAA", created)
    second = paths.capture_path("c_01HQBBBBBB", created)
    assert first != second


def test_note_path_combines_id_and_slug(tmp_path):
    paths = VaultPaths(tmp_path)
    path = paths.note_path("n_01hq", "Scaling Plateau")
    assert path == tmp_path / "notes" / "n_01hq-scaling-plateau.md"


def test_entity_and_community_paths(tmp_path):
    paths = VaultPaths(tmp_path)
    assert paths.entity_path("scaling-laws") == tmp_path / "entities" / "scaling-laws.md"
    assert paths.community_path("g_01ab") == tmp_path / "communities" / "g_01ab.md"


def test_community_path_is_stable_across_retitling(tmp_path):
    """Titles change; lineage ids do not. A title in the filename would leave a
    second file behind on every retitle, and read_report would glob arbitrarily."""
    paths = VaultPaths(tmp_path)
    assert paths.community_path("g_01ab") == paths.community_path("g_01ab")


def test_well_known_locations(tmp_path):
    paths = VaultPaths(tmp_path)
    assert paths.mindpalace_md == tmp_path / "MINDPALACE.md"
    assert paths.graph_db == tmp_path / ".graph" / "mindpalace.db"
    assert paths.decisions_log == tmp_path / ".mindpalace" / "decisions.jsonl"
    assert paths.op_log == tmp_path / ".mindpalace" / "log.jsonl"
    assert paths.lock == tmp_path / ".mindpalace" / "lock"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_vault_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.vault'`

- [ ] **Step 3: Implement paths**

```bash
mkdir -p mindpalace/vault
touch mindpalace/vault/__init__.py
```

`mindpalace/vault/paths.py`:

```python
"""Canonical on-disk layout for a Mind Palace vault."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mindpalace.ids import slugify

SUFFIX_LENGTH = 6


class VaultPaths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.mindpalace_md = self.root / "MINDPALACE.md"
        self.index_md = self.root / "index.md"
        self.captures = self.root / "captures"
        self.notes = self.root / "notes"
        self.entities = self.root / "entities"
        self.communities = self.root / "communities"
        self.graph_db = self.root / ".graph" / "mindpalace.db"
        self.decisions_log = self.root / ".mindpalace" / "decisions.jsonl"
        self.op_log = self.root / ".mindpalace" / "log.jsonl"
        self.lock = self.root / ".mindpalace" / "lock"

    def capture_path(self, capture_id: str, created: datetime) -> Path:
        suffix = capture_id[-SUFFIX_LENGTH:]
        return self.captures / f"{created:%Y-%m-%d-%H%M}-{suffix}.md"

    def note_path(self, note_id: str, slug: str) -> Path:
        return self.notes / f"{note_id}-{slugify(slug)}.md"

    def entity_path(self, slug: str) -> Path:
        return self.entities / f"{slugify(slug)}.md"

    def community_path(self, lineage_id: str) -> Path:
        """Keyed by lineage id alone. Including the title would mint a new file
        every time a report is retitled, leaving orphans behind."""
        return self.communities / f"{lineage_id}.md"

    def all_directories(self) -> list[Path]:
        return [
            self.captures,
            self.notes,
            self.entities,
            self.communities,
            self.graph_db.parent,
            self.op_log.parent,
        ]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_vault_paths.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the failing store test**

`tests/test_vault_store.py`:

```python
from datetime import UTC, datetime

import pytest

from mindpalace.atomic import ConflictError
from mindpalace.models import Capture, CommunityReport, EntityPage, Note
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


def make_capture(capture_id="c_01HQABCDEF"):
    return Capture(
        id=capture_id,
        created="2026-08-08T14:22:00Z",
        source="manual",
        why=None,
        text="The plateau talk is about data exhaustion.",
    )


def test_capture_write_then_read(store):
    capture = make_capture()
    path = store.write_capture(capture, datetime(2026, 8, 8, 14, 22, tzinfo=UTC))
    assert store.read_capture(path) == capture


def test_capture_write_refuses_to_overwrite(store):
    capture = make_capture()
    when = datetime(2026, 8, 8, 14, 22, tzinfo=UTC)
    store.write_capture(capture, when)
    with pytest.raises(ConflictError):
        store.write_capture(capture, when)


def test_note_write_then_read_and_iterate(store):
    note = Note(
        id="n_01hq",
        derived_from="c_01hq",
        created="2026-08-08T14:25:00Z",
        author="llm",
        body="Analysis.",
    )
    store.write_note(note, "scaling plateau")
    assert list(store.iter_notes()) == [note]


def test_entity_page_round_trips_with_user_block_and_related(store):
    page = EntityPage(
        slug="scaling-laws",
        type="concept",
        description="Compute, data, and loss.",
        generated_from=["n_01hq"],
        input_hash="sha256:abc",
        stale=False,
        user={"aliases": ["scaling law"]},
        related=["contradicts [[data-exhaustion]]"],
    )
    store.write_entity_page(page)
    assert store.read_entity_page("scaling-laws") == page


def test_read_missing_entity_page_returns_none(store):
    assert store.read_entity_page("nobody") is None


def test_report_round_trips(store):
    report = CommunityReport(
        lineage_id="g_01ab",
        level=0,
        title="Scaling Debate",
        summary="A cluster about scaling limits.",
        rank=6.5,
        findings=[{"summary": "Data supply dominates", "explanation": "…"}],
        cites=["e_scaling-laws", "x_01ab"],
        generated_from=["e_scaling-laws"],
        input_hash="sha256:def",
        stale=False,
    )
    store.write_report(report)
    assert store.read_report("g_01ab") == report


def test_retitling_a_report_does_not_leave_a_second_file(store):
    for title in ("First Title", "Second Title"):
        store.write_report(
            CommunityReport(
                lineage_id="g_01ab",
                level=0,
                title=title,
                summary="Same community, renamed.",
                rank=5.0,
            )
        )
    assert len(list(store.paths.communities.glob("*.md"))) == 1
    assert store.read_report("g_01ab").title == "Second Title"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_vault_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.vault.store'`

- [ ] **Step 7: Implement the store**

`mindpalace/vault/store.py`:

```python
"""Tiered read/write access to the vault.

Tier 1 (captures, notes) is written once and never mutated. Tier 2 (entity
pages, community reports) is rewritten in place but never deleted by rebuild.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from mindpalace.atomic import atomic_write, cas_write
from mindpalace.frontmatter import parse, render
from mindpalace.models import (
    Capture,
    CommunityReport,
    EntityPage,
    Note,
    capture_from_markdown,
    capture_to_markdown,
    note_from_markdown,
    note_to_markdown,
)
from mindpalace.vault.paths import VaultPaths

RELATED_OPEN = "<!-- mindpalace:related -->"
RELATED_CLOSE = "<!-- /mindpalace:related -->"
GENERATED_BANNER = (
    "<!-- Machine-written. Edits below are replaced on regeneration; "
    "put durable changes under `user:` in the front-matter. -->"
)


class VaultStore:
    def __init__(self, paths: VaultPaths) -> None:
        self.paths = paths

    # ---- Tier 1 -------------------------------------------------------

    def write_capture(self, capture: Capture, created: datetime) -> Path:
        path = self.paths.capture_path(capture.id, created)
        cas_write(path, capture_to_markdown(capture), None)
        return path

    def read_capture(self, path: Path) -> Capture:
        return capture_from_markdown(path.read_text(encoding="utf-8"))

    def iter_captures(self) -> Iterator[Capture]:
        for path in sorted(self.paths.captures.glob("*.md")):
            yield self.read_capture(path)

    def write_note(self, note: Note, slug: str) -> Path:
        path = self.paths.note_path(note.id, slug)
        cas_write(path, note_to_markdown(note), None)
        return path

    def read_note(self, path: Path) -> Note:
        return note_from_markdown(path.read_text(encoding="utf-8"))

    def iter_notes(self) -> Iterator[Note]:
        for path in sorted(self.paths.notes.glob("*.md")):
            yield self.read_note(path)

    # ---- Tier 2 -------------------------------------------------------

    def write_entity_page(self, page: EntityPage) -> Path:
        data = {
            "id": f"e_{page.slug}",
            "type": page.type,
            "generated_from": page.generated_from,
            "input_hash": page.input_hash,
            "stale": page.stale,
        }
        if page.user:
            data["user"] = page.user
        related = "\n".join(f"- {line}" for line in page.related)
        body = (
            f"{GENERATED_BANNER}\n\n{page.description}\n\n"
            f"{RELATED_OPEN}\n{related}\n{RELATED_CLOSE}"
        )
        path = self.paths.entity_path(page.slug)
        atomic_write(path, render(data, body))
        return path

    def read_entity_page(self, slug: str) -> EntityPage | None:
        path = self.paths.entity_path(slug)
        if not path.exists():
            return None
        data, body = parse(path.read_text(encoding="utf-8"))
        description, related = _split_related(body)
        return EntityPage(
            slug=slug,
            type=data["type"],
            description=description,
            generated_from=data.get("generated_from", []),
            input_hash=data.get("input_hash", ""),
            stale=data.get("stale", True),
            user=data.get("user", {}),
            related=related,
        )

    def iter_entity_pages(self) -> Iterator[EntityPage]:
        for path in sorted(self.paths.entities.glob("*.md")):
            page = self.read_entity_page(path.stem)
            if page is not None:
                yield page

    def write_report(self, report: CommunityReport) -> Path:
        data = {
            "lineage_id": report.lineage_id,
            "level": report.level,
            "title": report.title,
            "rank": report.rank,
            "cites": report.cites,
            "generated_from": report.generated_from,
            "input_hash": report.input_hash,
            "stale": report.stale,
            "findings": report.findings,
        }
        path = self.paths.community_path(report.lineage_id)
        atomic_write(path, render(data, f"{GENERATED_BANNER}\n\n{report.summary}"))
        return path

    def read_report(self, lineage_id: str) -> CommunityReport | None:
        path = self.paths.community_path(lineage_id)
        if not path.exists():
            return None
        data, body = parse(path.read_text(encoding="utf-8"))
        return CommunityReport(
            lineage_id=data["lineage_id"],
            level=data["level"],
            title=data["title"],
            summary=_strip_banner(body),
            rank=data["rank"],
            findings=data.get("findings", []),
            cites=data.get("cites", []),
            generated_from=data.get("generated_from", []),
            input_hash=data.get("input_hash", ""),
            stale=data.get("stale", False),
        )

    def iter_reports(self) -> Iterator[CommunityReport]:
        for path in sorted(self.paths.communities.glob("*.md")):
            report = self.read_report(path.stem)
            if report is not None:
                yield report


def _strip_banner(body: str) -> str:
    return body.replace(GENERATED_BANNER, "", 1).strip()


def _split_related(body: str) -> tuple[str, list[str]]:
    text = _strip_banner(body)
    if RELATED_OPEN not in text:
        return text.strip(), []
    description, _, remainder = text.partition(RELATED_OPEN)
    block, _, _ = remainder.partition(RELATED_CLOSE)
    related = [
        line.removeprefix("- ").strip()
        for line in block.strip().splitlines()
        if line.strip()
    ]
    return description.strip(), related
```

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_vault_store.py -v`
Expected: PASS (7 tests)

- [ ] **Step 9: Commit**

```bash
git add mindpalace/vault/ tests/test_vault_paths.py tests/test_vault_store.py
git commit -m "feat: vault layout and tiered markdown store"
```

---

## Task 5: `MINDPALACE.md` config, scaffolding, and the vault contract

**Files:**
- Create: `mindpalace/config.py`, `mindpalace/templates/MINDPALACE.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `mindpalace.frontmatter.parse`, `mindpalace.vault.paths.VaultPaths`, `mindpalace.atomic.atomic_write`
- Produces: dataclasses `EdgeType(name, directed, cluster_weight)`, `Thresholds(cluster_activation_entities, abstain_bm25_floor, abstain_cosine_floor, community_lineage_jaccard)`, `Config(schema_version, entity_types, edge_types, thresholds, embedder, templates)`; `ConfigError`; `load_config(path) -> Config`; `default_config_text() -> str`; `scaffold(paths) -> None`; `open_vault(root, *, init=False) -> tuple[VaultPaths, Config]`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
import pytest

from mindpalace.config import (
    ConfigError,
    default_config_text,
    load_config,
    open_vault,
    scaffold,
)
from mindpalace.vault.paths import VaultPaths

MINIMAL = """---
schema_version: 1
entity_types: [concept]
edge_types:
  relates-to: {directed: false, cluster_weight: 1.0}
  supports: {directed: true, cluster_weight: 1.0}
thresholds:
  cluster_activation_entities: 150
  abstain_bm25_floor: 2.0
  abstain_cosine_floor: 0.35
  community_lineage_jaccard: 0.5
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Do the thing.

## template: report_next
Write the report.
"""


def write_config(tmp_path, text):
    path = tmp_path / "MINDPALACE.md"
    path.write_text(text)
    return path


def test_load_parses_structure_and_templates(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL))
    assert config.schema_version == 1
    assert config.entity_types == ["concept"]
    assert config.edge_types["relates-to"].directed is False
    assert config.edge_types["supports"].directed is True
    assert config.edge_types["relates-to"].cluster_weight == 1.0
    assert config.thresholds.cluster_activation_entities == 150
    assert config.thresholds.abstain_cosine_floor == 0.35
    assert config.embedder == {"kind": "local", "model": "BAAI/bge-small-en-v1.5"}
    assert config.templates["extraction_next"] == "Do the thing."
    assert config.templates["report_next"] == "Write the report."


def test_missing_required_key_names_the_key_and_file(tmp_path):
    broken = MINIMAL.replace("entity_types: [concept]\n", "")
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "entity_types" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    broken = MINIMAL.replace("schema_version: 1", "schema_version: 1\nmystery: 3")
    with pytest.raises(ConfigError, match="mystery"):
        load_config(write_config(tmp_path, broken))


def test_unsupported_schema_version_is_rejected(tmp_path):
    broken = MINIMAL.replace("schema_version: 1", "schema_version: 99")
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(write_config(tmp_path, broken))


def test_edge_type_missing_directed_is_rejected(tmp_path):
    broken = MINIMAL.replace(
        "relates-to: {directed: false, cluster_weight: 1.0}",
        "relates-to: {cluster_weight: 1.0}",
    )
    with pytest.raises(ConfigError, match="directed"):
        load_config(write_config(tmp_path, broken))


def test_default_config_is_itself_valid(tmp_path):
    config = load_config(write_config(tmp_path, default_config_text()))
    assert set(config.edge_types) == {
        "relates-to",
        "supports",
        "contradicts",
        "example-of",
        "part-of",
        "derived-from",
        "mentions",
    }
    assert config.edge_types["contradicts"].directed is False
    assert config.edge_types["supports"].directed is True
    assert "extraction_next" in config.templates


def test_scaffold_creates_layout_and_config(tmp_path):
    paths = VaultPaths(tmp_path)
    scaffold(paths)
    assert paths.mindpalace_md.exists()
    for directory in paths.all_directories():
        assert directory.is_dir()


def test_open_vault_requires_init_for_empty_directory(tmp_path):
    with pytest.raises(ConfigError, match="--init"):
        open_vault(tmp_path)


def test_open_vault_with_init_scaffolds_then_loads(tmp_path):
    paths, config = open_vault(tmp_path, init=True)
    assert paths.mindpalace_md.exists()
    assert config.schema_version == 1


def test_open_vault_refuses_non_empty_directory_without_config(tmp_path):
    (tmp_path / "unrelated.txt").write_text("hello")
    with pytest.raises(ConfigError, match="not a Mind Palace vault"):
        open_vault(tmp_path, init=True)


def test_open_vault_refuses_home_and_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConfigError, match="refusing"):
        open_vault(tmp_path, init=True)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.config'`

- [ ] **Step 3: Write the default template**

```bash
mkdir -p mindpalace/templates
```

`mindpalace/templates/MINDPALACE.md`:

```markdown
---
schema_version: 1
entity_types: [person, concept, paper, project, term, theme]
edge_types:
  relates-to: {directed: false, cluster_weight: 1.0}
  contradicts: {directed: false, cluster_weight: 1.0}
  supports: {directed: true, cluster_weight: 1.0}
  example-of: {directed: true, cluster_weight: 1.0}
  part-of: {directed: true, cluster_weight: 1.0}
  derived-from: {directed: true, cluster_weight: 1.0}
  mentions: {directed: true, cluster_weight: 0.5}
thresholds:
  cluster_activation_entities: 150
  abstain_bm25_floor: 2.0
  abstain_cosine_floor: 0.35
  community_lineage_jaccard: 0.5
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Read any nearest notes you need, then call write_note with this capture id. The
note body must contain your analysis — claims, why it matters — not a
restatement of the capture. Extract entities with a type and a one-line
description. Propose a relationship only where you can give a specific rationale
citing both endpoints; give each a strength 1-10. Reuse an existing entity name
over inventing a near-duplicate. Check previously_dismissed before re-proposing a
pair, and only re-propose when your evidence genuinely differs. Proposing nothing
is a valid outcome.

## template: report_next
Write one report per community using write_community_report. Ground every finding
in the supplied members with an inline citation of the form
[Data: Entities (e_slug); Assertions (x_id)], and list the same ids in cites.
Do not state anything the supplied members do not support.

## template: abstain
Nothing in the vault is relevant to this query. Say so rather than answering from
your own knowledge.
```

- [ ] **Step 4: Implement config loading**

`mindpalace/config.py`:

```python
"""Parsing, validation, and scaffolding of MINDPALACE.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mindpalace.atomic import atomic_write
from mindpalace.frontmatter import FrontMatterError, parse
from mindpalace.vault.paths import VaultPaths

SUPPORTED_SCHEMA_VERSIONS = {1}
REQUIRED_KEYS = {
    "schema_version",
    "entity_types",
    "edge_types",
    "thresholds",
    "embedder",
}
REQUIRED_THRESHOLDS = {
    "cluster_activation_entities",
    "abstain_bm25_floor",
    "abstain_cosine_floor",
    "community_lineage_jaccard",
}
TEMPLATE_HEADING = re.compile(r"^##\s+template:\s*(\S+)\s*$", re.MULTILINE)


class ConfigError(ValueError):
    """Raised for any invalid or missing configuration."""


@dataclass(frozen=True)
class EdgeType:
    name: str
    directed: bool
    cluster_weight: float


@dataclass(frozen=True)
class Thresholds:
    cluster_activation_entities: int
    abstain_bm25_floor: float
    abstain_cosine_floor: float
    community_lineage_jaccard: float


@dataclass(frozen=True)
class Config:
    schema_version: int
    entity_types: list[str]
    edge_types: dict[str, EdgeType]
    thresholds: Thresholds
    embedder: dict
    templates: dict[str, str]

    def is_symmetric(self, edge_type: str) -> bool:
        try:
            return not self.edge_types[edge_type].directed
        except KeyError:
            raise ConfigError(f"unknown edge type {edge_type!r}") from None


def default_config_text() -> str:
    return (
        resources.files("mindpalace.templates")
        .joinpath("MINDPALACE.md")
        .read_text(encoding="utf-8")
    )


def load_config(path: Path) -> Config:
    try:
        data, body = parse(path.read_text(encoding="utf-8"))
    except FrontMatterError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ConfigError(f"{path.name}: missing required key(s) {sorted(missing)}")
    unknown = data.keys() - REQUIRED_KEYS
    if unknown:
        raise ConfigError(f"{path.name}: unknown key(s) {sorted(unknown)}")

    version = data["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"{path.name}: schema_version {version} is unsupported "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    edge_types = {}
    for name, spec in data["edge_types"].items():
        for field in ("directed", "cluster_weight"):
            if field not in spec:
                raise ConfigError(
                    f"{path.name}: edge type {name!r} is missing {field!r}"
                )
        edge_types[name] = EdgeType(
            name=name,
            directed=bool(spec["directed"]),
            cluster_weight=float(spec["cluster_weight"]),
        )

    raw_thresholds = data["thresholds"]
    missing_thresholds = REQUIRED_THRESHOLDS - raw_thresholds.keys()
    if missing_thresholds:
        raise ConfigError(
            f"{path.name}: thresholds missing {sorted(missing_thresholds)}"
        )

    return Config(
        schema_version=version,
        entity_types=list(data["entity_types"]),
        edge_types=edge_types,
        thresholds=Thresholds(**{k: raw_thresholds[k] for k in REQUIRED_THRESHOLDS}),
        embedder=dict(data["embedder"]),
        templates=_parse_templates(body),
    )


def _parse_templates(body: str) -> dict[str, str]:
    matches = list(TEMPLATE_HEADING.finditer(body))
    templates: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        templates[match.group(1)] = body[match.end() : end].strip()
    return templates


def scaffold(paths: VaultPaths) -> None:
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write(paths.mindpalace_md, default_config_text())
    atomic_write(paths.index_md, "# Index\n\n_No notes yet._\n")


def open_vault(root: Path, *, init: bool = False) -> tuple[VaultPaths, Config]:
    root = Path(root).expanduser().resolve()
    if root == Path.home() or root == Path(root.anchor):
        raise ConfigError(f"refusing to use {root} as a vault root")

    paths = VaultPaths(root)
    if paths.mindpalace_md.exists():
        return paths, load_config(paths.mindpalace_md)

    if not init:
        raise ConfigError(f"{root} has no MINDPALACE.md; re-run with --init")

    if root.exists() and any(root.iterdir()):
        raise ConfigError(
            f"{root} is not a Mind Palace vault and is not empty; refusing to --init"
        )

    scaffold(paths)
    return paths, load_config(paths.mindpalace_md)
```

Add the template to the wheel by appending to `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["mindpalace"]

[tool.hatch.build.targets.wheel.force-include]
"mindpalace/templates" = "mindpalace/templates"
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add mindpalace/config.py mindpalace/templates/ pyproject.toml tests/test_config.py
git commit -m "feat: MINDPALACE.md config, scaffolding, and vault-open contract"
```

---

## Task 6: Operation log, decision log, and crash replay

**Files:**
- Create: `mindpalace/oplog.py`
- Test: `tests/test_oplog.py`

**Interfaces:**
- Consumes: `mindpalace.ids.new_id`, `mindpalace.models.Decision`
- Produces: `OpLog(path)` with `begin(intent) -> str`, `commit(op_id) -> None`, `pending() -> list[dict]`; `DecisionLog(path)` with `append(assertion_id, action, via, op_id, reason=None) -> None`, `entries() -> list[Decision]`, `status_map() -> dict[str, str]`, `dismissal_reasons() -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

`tests/test_oplog.py`:

```python
from mindpalace.oplog import DecisionLog, OpLog


def test_begin_marks_pending_and_commit_clears_it(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture", "capture": "c_01"})
    assert len(log.pending()) == 1
    log.commit(op_id)
    assert log.pending() == []


def test_begin_without_commit_is_pending(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture", "capture": "c_01"})
    reopened = OpLog(tmp_path / "log.jsonl")
    pending = reopened.pending()
    assert [entry["op"] for entry in pending] == [op_id]
    assert pending[0]["intent"]["tool"] == "save_capture"


def test_only_uncommitted_operations_are_pending(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    done = log.begin({"tool": "save_capture"})
    log.commit(done)
    stranded = log.begin({"tool": "write_note"})
    assert [entry["op"] for entry in OpLog(tmp_path / "log.jsonl").pending()] == [
        stranded
    ]


def test_commit_is_idempotent(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture"})
    log.commit(op_id)
    log.commit(op_id)
    assert log.pending() == []


def test_status_map_is_proposed_until_a_decision_exists(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    assert decisions.status_map() == {}


def test_status_map_takes_the_latest_decision(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "wrong sense")
    decisions.append("x_01", "confirm", "resolve_assertion", "op_2", "new evidence")
    assert decisions.status_map() == {"x_01": "confirm"}


def test_decisions_survive_reopen(tmp_path):
    path = tmp_path / "decisions.jsonl"
    DecisionLog(path).append("x_01", "confirm", "resolve_assertion", "op_1")
    assert DecisionLog(path).status_map() == {"x_01": "confirm"}


def test_dismissal_reasons_are_recoverable(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "different sense")
    assert decisions.dismissal_reasons() == {"x_01": "different sense"}


def test_confirmation_clears_a_prior_dismissal_reason(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "different sense")
    decisions.append("x_01", "confirm", "resolve_assertion", "op_2")
    assert decisions.dismissal_reasons() == {}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_oplog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.oplog'`

- [ ] **Step 3: Implement both logs**

`mindpalace/oplog.py`:

```python
"""Append-only logs: write-ahead operations and durable review decisions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from mindpalace.ids import new_id
from mindpalace.models import Decision


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class OpLog:
    """Write-ahead log. An op with a begin and no commit is replayed on startup."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def begin(self, intent: dict) -> str:
        op_id = new_id("op_")
        _append_line(
            self.path, {"kind": "op.begin", "op": op_id, "ts": _now(), "intent": intent}
        )
        return op_id

    def commit(self, op_id: str) -> None:
        _append_line(self.path, {"kind": "op.commit", "op": op_id, "ts": _now()})

    def pending(self) -> list[dict]:
        begun: dict[str, dict] = {}
        for record in _read_lines(self.path):
            if record["kind"] == "op.begin":
                begun[record["op"]] = record
            elif record["kind"] == "op.commit":
                begun.pop(record["op"], None)
        return list(begun.values())


class DecisionLog:
    """Durable review events. Assertion status is the fold of this log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(
        self,
        assertion_id: str,
        action: str,
        via: str,
        op_id: str,
        reason: str | None = None,
    ) -> None:
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": _now(),
                "assertion": assertion_id,
                "action": action,
                "via": via,
                "reason": reason,
            },
        )

    def entries(self) -> list[Decision]:
        return [
            Decision(
                op=record["op"],
                ts=record["ts"],
                assertion=record["assertion"],
                action=record["action"],
                via=record["via"],
                reason=record.get("reason"),
            )
            for record in _read_lines(self.path)
        ]

    def status_map(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for decision in self.entries():
            statuses[decision.assertion] = decision.action
        return statuses

    def dismissal_reasons(self) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for decision in self.entries():
            if decision.action == "dismiss":
                reasons[decision.assertion] = decision.reason or ""
            else:
                reasons.pop(decision.assertion, None)
        return reasons
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_oplog.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/oplog.py tests/test_oplog.py
git commit -m "feat: write-ahead operation log and durable decision log"
```

---

## Task 7: SQLite cache schema and vector storage

**Files:**
- Create: `mindpalace/index/__init__.py`, `mindpalace/index/db.py`, `mindpalace/index/vectors.py`
- Test: `tests/test_index_db.py`, `tests/test_vectors.py`

**Interfaces:**
- Consumes: nothing
- Produces: `db.connect(path) -> sqlite3.Connection`, `db.create_schema(conn) -> None`, `db.TABLES: frozenset[str]`; `vectors.store(conn, doc_id, kind, model_id, vector) -> None`, `vectors.search(conn, query_vector, model_id, kinds, limit) -> list[tuple[str, float]]`, `vectors.clear(conn) -> None`

Everything in this task is Tier 3 — deletable and rebuildable. Nothing here is
ever the source of truth.

- [ ] **Step 1: Write the failing schema test**

`tests/test_index_db.py`:

```python
from mindpalace.index import db


def test_create_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    db.create_schema(conn)  # must not raise
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert db.TABLES <= names


def test_fts_table_supports_match(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
        ("n_01", "note", "Scaling", "the plateau is about data exhaustion"),
    )
    rows = list(
        conn.execute(
            "SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY rank",
            ("exhaustion",),
        )
    )
    assert [row[0] for row in rows] == ["n_01"]


def test_foreign_keys_and_wal_are_enabled(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_file_hashes_round_trip(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO files (path, hash) VALUES (?, ?)", ("notes/n_01.md", "sha256:aa")
    )
    conn.execute(
        "INSERT INTO files (path, hash) VALUES (?, ?) "
        "ON CONFLICT(path) DO UPDATE SET hash = excluded.hash",
        ("notes/n_01.md", "sha256:bb"),
    )
    assert conn.execute("SELECT hash FROM files").fetchone()[0] == "sha256:bb"


def test_cache_meta_is_absent_until_written(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    assert db.read_meta(conn) is None


def test_cache_meta_round_trips_and_upserts(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    db.create_schema(conn)
    db.write_meta(conn, "stub-64", 64)
    assert db.read_meta(conn) == ("stub-64", 64)
    db.write_meta(conn, "bge-small", 384)
    assert db.read_meta(conn) == ("bge-small", 384)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_index_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.index'`

- [ ] **Step 3: Implement the schema**

```bash
mkdir -p mindpalace/index
touch mindpalace/index/__init__.py
```

`mindpalace/index/db.py`:

```python
"""Tier 3 SQLite cache. Deletable and fully rebuildable from Tier 1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TABLES = frozenset(
    {
        "cache_meta",
        "files",
        "entities",
        "entity_sources",
        "assertions",
        "aggregates",
        "aggregate_members",
        "claims",
        "communities",
        "docs",
        "vectors",
        "vault_issues",
    }
)

SCHEMA = """
-- Single-row table describing what produced this cache. A mismatch against the
-- live embedder forces a rebuild (spec §5) instead of silently returning zero
-- vector results, which is what a changed model would otherwise cause.
CREATE TABLE IF NOT EXISTS cache_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    slug TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entity_sources (
    slug TEXT NOT NULL,
    note_id TEXT NOT NULL,
    PRIMARY KEY (slug, note_id)
);

CREATE TABLE IF NOT EXISTS assertions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    note_id TEXT NOT NULL,
    source TEXT,
    target TEXT,
    type TEXT,
    strength INTEGER,
    subject TEXT,
    text TEXT,
    description TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aggregates (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    target TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    mean_strength REAL NOT NULL DEFAULT 0,
    traversable INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS aggregate_members (
    key TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    PRIMARY KEY (key, assertion_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS communities (
    lineage_id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    parent TEXT,
    members TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    doc_id UNINDEXED,
    kind UNINDEXED,
    title,
    text
);

CREATE TABLE IF NOT EXISTS vectors (
    doc_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_issues (
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aggregates_source ON aggregates(source);
CREATE INDEX IF NOT EXISTS idx_aggregates_target ON aggregates(target);
CREATE INDEX IF NOT EXISTS idx_assertions_note ON assertions(note_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def read_meta(conn: sqlite3.Connection) -> tuple[str, int] | None:
    row = conn.execute("SELECT model_id, dim FROM cache_meta WHERE id = 1").fetchone()
    return None if row is None else (row["model_id"], row["dim"])


def write_meta(conn: sqlite3.Connection, model_id: str, dim: int) -> None:
    conn.execute(
        "INSERT INTO cache_meta (id, model_id, dim) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id, "
        "dim = excluded.dim",
        (model_id, dim),
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_index_db.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing vectors test**

`tests/test_vectors.py`:

```python
import numpy as np
import pytest

from mindpalace.index import db, vectors


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "cache.db")
    db.create_schema(connection)
    return connection


def test_search_ranks_by_cosine_similarity(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0, 0.0]))
    vectors.store(conn, "n_02", "note", "stub", np.array([0.0, 1.0, 0.0]))
    results = vectors.search(conn, np.array([1.0, 0.1, 0.0]), "stub", ["note"], 10)
    assert [doc_id for doc_id, _ in results] == ["n_01", "n_02"]
    assert results[0][1] > results[1][1]


def test_cosine_is_magnitude_invariant(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([3.0, 0.0, 0.0]))
    [(_, score)] = vectors.search(conn, np.array([9.0, 0.0, 0.0]), "stub", ["note"], 10)
    assert score == pytest.approx(1.0)


def test_search_filters_by_kind(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.store(conn, "e_x", "entity", "stub", np.array([1.0, 0.0]))
    results = vectors.search(conn, np.array([1.0, 0.0]), "stub", ["entity"], 10)
    assert [doc_id for doc_id, _ in results] == ["e_x"]


def test_search_ignores_other_models(conn):
    vectors.store(conn, "n_01", "note", "other-model", np.array([1.0, 0.0]))
    assert vectors.search(conn, np.array([1.0, 0.0]), "stub", ["note"], 10) == []


def test_store_replaces_existing_vector(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.store(conn, "n_01", "note", "stub", np.array([0.0, 1.0]))
    [(_, score)] = vectors.search(conn, np.array([0.0, 1.0]), "stub", ["note"], 10)
    assert score == pytest.approx(1.0)


def test_limit_is_respected(conn):
    for index in range(5):
        vectors.store(conn, f"n_{index}", "note", "stub", np.array([1.0, float(index)]))
    assert len(vectors.search(conn, np.array([1.0, 1.0]), "stub", ["note"], 2)) == 2


def test_clear_removes_everything(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.clear(conn)
    assert vectors.search(conn, np.array([1.0, 0.0]), "stub", ["note"], 10) == []
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_vectors.py -v`
Expected: FAIL with `ImportError: cannot import name 'vectors'`

- [ ] **Step 7: Implement vector storage**

`mindpalace/index/vectors.py`:

```python
"""Vector storage and brute-force cosine search.

Deliberate deviation from the spec's `sqlite-vec`: at MVP scale a numpy scan
over a few hundred vectors is sub-millisecond, and this avoids a loadable
extension that is fragile across macOS Python builds. Swapping in `sqlite-vec`
later touches only this module.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

DTYPE = np.float32


def _normalise(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=DTYPE).ravel()
    norm = float(np.linalg.norm(array))
    return array if norm == 0.0 else array / norm


def store(
    conn: sqlite3.Connection,
    doc_id: str,
    kind: str,
    model_id: str,
    vector: np.ndarray,
) -> None:
    unit = _normalise(vector)
    conn.execute(
        "INSERT INTO vectors (doc_id, kind, model_id, dim, vector) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET kind=excluded.kind, "
        "model_id=excluded.model_id, dim=excluded.dim, vector=excluded.vector",
        (doc_id, kind, model_id, int(unit.size), unit.tobytes()),
    )


def search(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    model_id: str,
    kinds: Sequence[str],
    limit: int,
) -> list[tuple[str, float]]:
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT doc_id, vector FROM vectors "
        f"WHERE model_id = ? AND kind IN ({placeholders})",
        (model_id, *kinds),
    ).fetchall()
    if not rows:
        return []

    query = _normalise(query_vector)
    matrix = np.vstack([np.frombuffer(row["vector"], dtype=DTYPE) for row in rows])
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[:limit]
    return [(rows[index]["doc_id"], float(scores[index])) for index in order]


def clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM vectors")
```

**Neither helper commits.** The caller owns the transaction — `sync` wipes and
repopulates the whole cache, and a `commit()` inside a helper would end that
transaction midway, committing the wipe independently of the repopulation. The
tests above still pass unchanged because uncommitted writes are visible on the
connection that made them; only cross-connection reads would need a commit.

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_vectors.py -v`
Expected: PASS (7 tests)

- [ ] **Step 9: Commit**

```bash
git add mindpalace/index/ tests/test_index_db.py tests/test_vectors.py
git commit -m "feat: SQLite cache schema and brute-force vector search"
```

---

## Task 8: Embedder protocol, stub, and local model

**Files:**
- Create: `mindpalace/embed.py`
- Test: `tests/test_embed.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `mindpalace.config.Config`
- Produces: `Embedder` protocol (`model_id: str`, `dim: int`, `embed(texts) -> np.ndarray`), `StubEmbedder(dim=64)`, `LocalEmbedder(model_name)`, `get_embedder(config) -> Embedder`, `EmbedderError`

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:

```python
import numpy as np
import pytest

from mindpalace.embed import EmbedderError, StubEmbedder, get_embedder


def test_stub_is_deterministic():
    first = StubEmbedder().embed(["hello world"])
    second = StubEmbedder().embed(["hello world"])
    assert np.array_equal(first, second)


def test_stub_distinguishes_texts():
    embedder = StubEmbedder()
    vectors = embedder.embed(["hello world", "entirely different"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_stub_shape_matches_declared_dim():
    embedder = StubEmbedder(dim=32)
    assert embedder.embed(["a", "b"]).shape == (2, 32)
    assert embedder.dim == 32


def test_stub_vectors_are_unit_length():
    vectors = StubEmbedder().embed(["hello"])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-6)


def test_stub_handles_empty_input():
    assert StubEmbedder().embed([]).shape == (0, 64)


def test_get_embedder_returns_stub_for_stub_kind():
    embedder = get_embedder({"kind": "stub", "dim": 16})
    assert embedder.model_id == "stub-16"


def test_get_embedder_rejects_unknown_kind():
    with pytest.raises(EmbedderError, match="unknown embedder kind"):
        get_embedder({"kind": "telepathy"})


@pytest.mark.network
def test_local_embedder_produces_real_vectors():
    from mindpalace.embed import LocalEmbedder

    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    vectors = embedder.embed(["scaling laws", "banana bread"])
    assert vectors.shape == (2, embedder.dim)
    assert float(vectors[0] @ vectors[1]) < 0.9
```

`tests/conftest.py` — create the file with only the network-marker guard. Tests
construct `StubEmbedder()` directly where they need one, so no shared embedder
fixture is defined; an unused fixture is dead code.

```python
"""Shared pytest configuration.

Every test uses StubEmbedder: no downloads, no network, fully deterministic.
The one test that exercises the real model is marked `network` and deselected
by default via `addopts` in pyproject.toml.
"""
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.embed'`

- [ ] **Step 3: Implement embedders**

`mindpalace/embed.py`:

```python
"""Pluggable embedding backends. Local by default; stub in tests."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

DTYPE = np.float32


class EmbedderError(RuntimeError):
    """Raised for an unknown or unusable embedder configuration."""


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class StubEmbedder:
    """Deterministic hash-derived vectors. Never touches the network."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model_id = f"stub-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=DTYPE)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            generator = np.random.default_rng(seed)
            vector = generator.standard_normal(self.dim).astype(DTYPE)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


class LocalEmbedder:
    """On-device embeddings via fastembed. Downloads the model on first use."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise EmbedderError(
                "fastembed is not installed; run `uv sync` to restore dependencies"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.model_id = model_name
        probe = np.array(list(self._model.embed(["dimension probe"]))[0])
        self.dim = int(probe.size)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=DTYPE)
        vectors = np.vstack([np.asarray(v, dtype=DTYPE) for v in self._model.embed(texts)])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def get_embedder(spec: dict) -> Embedder:
    kind = spec.get("kind")
    if kind == "stub":
        return StubEmbedder(dim=int(spec.get("dim", 64)))
    if kind == "local":
        return LocalEmbedder(spec["model"])
    if kind == "cloud":
        raise EmbedderError(
            "the cloud embedder is not implemented in v0; every capture and note "
            "body would be sent to a third party, so it needs an explicit build"
        )
    raise EmbedderError(f"unknown embedder kind {kind!r}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_embed.py -v`
Expected: PASS (7 tests, 1 deselected by the `network` marker)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/embed.py tests/test_embed.py tests/conftest.py
git commit -m "feat: embedder protocol with deterministic stub and local backend"
```

---

## Task 9: The fold — assertions plus decisions to graph tables

**Files:**
- Create: `mindpalace/graph/__init__.py`, `mindpalace/graph/fold.py`
- Test: `tests/test_fold.py`

**Interfaces:**
- Consumes: `mindpalace.ids.{slugify, aggregate_key}`, `mindpalace.models.Note`, `mindpalace.config.Config`
- Produces: dataclasses `FoldedEntity(slug, type, rank, note_ids)`, `Aggregate(key, source, type, target, weight, mean_strength, assertion_ids, traversable)`, `FoldedAssertion(id, note_id, source, target, type, strength, description, status)`, `FoldedClaim(id, note_id, subject, text, status)`, `GraphTables(entities, aggregates, assertions, claims)`; `fold(notes, statuses, config) -> GraphTables`; `ACTION_TO_STATUS: dict[str, str]`

**This is the most load-bearing module in the project.** It is a pure function
over the *complete* current note set plus the decision log. It never mutates
incrementally, which is precisely why re-processing an edited file cannot
double-count.

- [ ] **Step 1: Write the failing test**

`tests/test_fold.py`:

```python
import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.graph.fold import fold
from mindpalace.models import ClaimAssertion, EntityInstance, Note, RelationshipAssertion


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0),
            "supports": EdgeType("supports", directed=True, cluster_weight=1.0),
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )


def make_note(note_id, created, *, entities=(), relationships=(), claims=()):
    return Note(
        id=note_id,
        derived_from=f"c_{note_id[2:]}",
        created=created,
        author="llm",
        body="Body.",
        entities=tuple(entities),
        relationship_assertions=tuple(relationships),
        claim_assertions=tuple(claims),
    )


def rel(assertion_id, source, target, edge_type="contradicts", strength=5):
    return RelationshipAssertion(
        id=assertion_id,
        source=source,
        target=target,
        type=edge_type,
        strength=strength,
        description="because.",
    )


def test_empty_input_yields_empty_tables(config):
    tables = fold([], {}, config)
    assert tables.entities == {}
    assert tables.aggregates == {}


def test_entities_come_from_declared_instances(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        entities=[EntityInstance("Scaling Laws", "concept", "…")],
    )
    tables = fold([note], {}, config)
    assert tables.entities["scaling-laws"].type == "concept"
    assert tables.entities["scaling-laws"].note_ids == ("n_01",)


def test_assertion_endpoints_create_entities_with_unknown_type(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    tables = fold([note], {}, config)
    assert tables.entities["a"].type == "unknown"
    assert tables.entities["b"].type == "unknown"


def test_most_recent_note_wins_the_entity_type(config):
    older = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        entities=[EntityInstance("thing", "concept", "…")],
    )
    newer = make_note(
        "n_02",
        "2026-08-05T00:00:00Z",
        entities=[EntityInstance("thing", "paper", "…")],
    )
    assert fold([older, newer], {}, config).entities["thing"].type == "paper"
    assert fold([newer, older], {}, config).entities["thing"].type == "paper"


def test_symmetric_assertions_merge_into_one_aggregate(config):
    forward = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    reverse = make_note(
        "n_02", "2026-08-02T00:00:00Z", relationships=[rel("x_2", "b", "a")]
    )
    tables = fold([forward, reverse], {}, config)
    assert len(tables.aggregates) == 1
    [aggregate] = tables.aggregates.values()
    assert set(aggregate.assertion_ids) == {"x_1", "x_2"}


def test_directed_assertions_do_not_merge(config):
    forward = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="supports")],
    )
    reverse = make_note(
        "n_02",
        "2026-08-02T00:00:00Z",
        relationships=[rel("x_2", "b", "a", edge_type="supports")],
    )
    assert len(fold([forward, reverse], {}, config).aggregates) == 2


def test_aggregate_is_not_traversable_while_only_proposed(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    [aggregate] = fold([note], {}, config).aggregates.values()
    assert aggregate.traversable is False
    assert aggregate.weight == 0


def test_one_confirmation_makes_the_aggregate_traversable(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b", strength=8)]
    )
    [aggregate] = fold([note], {"x_1": "confirm"}, config).aggregates.values()
    assert aggregate.traversable is True
    assert aggregate.weight == 1
    assert aggregate.mean_strength == 8.0


def test_dismissing_one_of_two_leaves_the_aggregate_alive(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", strength=4), rel("x_2", "a", "b", strength=8)],
    )
    statuses = {"x_1": "dismiss", "x_2": "confirm"}
    [aggregate] = fold([note], statuses, config).aggregates.values()
    assert aggregate.traversable is True
    assert aggregate.weight == 1
    assert aggregate.mean_strength == 8.0


def test_rank_counts_only_traversable_degree(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[
            rel("x_1", "hub", "spoke-one"),
            rel("x_2", "hub", "spoke-two"),
        ],
    )
    tables = fold([note], {"x_1": "confirm"}, config)
    assert tables.entities["hub"].rank == 1
    assert tables.entities["spoke-one"].rank == 1
    assert tables.entities["spoke-two"].rank == 0


def test_claims_take_status_from_the_decision_log(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        claims=[ClaimAssertion("k_1", "a", "A claim."), ClaimAssertion("k_2", "a", "B.")],
    )
    tables = fold([note], {"k_1": "confirm"}, config)
    assert tables.claims["k_1"].status == "confirmed"
    assert tables.claims["k_2"].status == "proposed"


def test_fold_is_idempotent(config):
    notes = [
        make_note(
            "n_01",
            "2026-08-01T00:00:00Z",
            entities=[EntityInstance("a", "concept", "…")],
            relationships=[rel("x_1", "a", "b")],
        )
    ]
    assert fold(notes, {"x_1": "confirm"}, config) == fold(
        notes, {"x_1": "confirm"}, config
    )


def test_editing_a_note_retracts_its_old_assertions(config):
    """The guard against double-counting: fold is over the current set, not a delta."""
    version_one = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    version_two = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_2", "a", "c")]
    )
    after = fold([version_two], {"x_2": "confirm"}, config)
    assert "x_1" not in after.assertions
    assert len(after.aggregates) == 1
    assert "b" not in after.entities


def test_unknown_edge_type_is_rejected(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="invented")],
    )
    with pytest.raises(Exception, match="invented"):
        fold([note], {}, config)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_fold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.graph'`

- [ ] **Step 3: Implement the fold**

```bash
mkdir -p mindpalace/graph
touch mindpalace/graph/__init__.py
```

`mindpalace/graph/fold.py`:

```python
"""Derive the graph from the complete current source set.

This module is a pure function. Given every note plus the decision log it
produces entities, aggregate relationships, assertions, and claims. It has no
incremental path by design: recomputation from scratch is what makes
re-processing an edited or re-detected file safe.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from mindpalace.config import Config
from mindpalace.ids import aggregate_key, slugify
from mindpalace.models import Note

ACTION_TO_STATUS = {"confirm": "confirmed", "dismiss": "dismissed"}
UNKNOWN_TYPE = "unknown"


@dataclass(frozen=True)
class FoldedEntity:
    slug: str
    type: str
    rank: int
    note_ids: tuple[str, ...]


@dataclass(frozen=True)
class FoldedAssertion:
    id: str
    note_id: str
    source: str
    target: str
    type: str
    strength: int
    description: str
    status: str


@dataclass(frozen=True)
class FoldedClaim:
    id: str
    note_id: str
    subject: str
    text: str
    status: str


@dataclass(frozen=True)
class Aggregate:
    key: str
    source: str
    type: str
    target: str
    weight: int
    mean_strength: float
    assertion_ids: tuple[str, ...]
    traversable: bool


@dataclass(frozen=True)
class GraphTables:
    entities: dict[str, FoldedEntity] = field(default_factory=dict)
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    assertions: dict[str, FoldedAssertion] = field(default_factory=dict)
    claims: dict[str, FoldedClaim] = field(default_factory=dict)


def _status(assertion_id: str, statuses: dict[str, str]) -> str:
    return ACTION_TO_STATUS.get(statuses.get(assertion_id, ""), "proposed")


def fold(
    notes: Iterable[Note], statuses: dict[str, str], config: Config
) -> GraphTables:
    ordered = sorted(notes, key=lambda note: (note.created, note.id))

    entity_types: dict[str, str] = {}
    entity_notes: dict[str, list[str]] = {}
    assertions: dict[str, FoldedAssertion] = {}
    claims: dict[str, FoldedClaim] = {}
    grouped: dict[str, list[FoldedAssertion]] = {}
    aggregate_shape: dict[str, tuple[str, str, str]] = {}

    def touch(slug: str, note_id: str, declared_type: str | None) -> None:
        if declared_type is not None or slug not in entity_types:
            entity_types[slug] = declared_type or UNKNOWN_TYPE
        note_ids = entity_notes.setdefault(slug, [])
        if note_id not in note_ids:
            note_ids.append(note_id)

    for note in ordered:
        for instance in note.entities:
            touch(slugify(instance.name), note.id, instance.type)

        for raw in note.relationship_assertions:
            if raw.type not in config.edge_types:
                raise ValueError(
                    f"note {note.id}: unknown edge type {raw.type!r} "
                    f"(assertion {raw.id})"
                )
            source, target = slugify(raw.source), slugify(raw.target)
            touch(source, note.id, None)
            touch(target, note.id, None)

            folded = FoldedAssertion(
                id=raw.id,
                note_id=note.id,
                source=source,
                target=target,
                type=raw.type,
                strength=raw.strength,
                description=raw.description,
                status=_status(raw.id, statuses),
            )
            assertions[raw.id] = folded

            key = aggregate_key(
                source, raw.type, target, symmetric=config.is_symmetric(raw.type)
            )
            grouped.setdefault(key, []).append(folded)
            if key not in aggregate_shape:
                left, _, rest = key.removeprefix("r:").partition("|")
                edge_type, _, right = rest.partition("|")
                aggregate_shape[key] = (left, edge_type, right)

        for raw_claim in note.claim_assertions:
            subject = slugify(raw_claim.subject)
            touch(subject, note.id, None)
            claims[raw_claim.id] = FoldedClaim(
                id=raw_claim.id,
                note_id=note.id,
                subject=subject,
                text=raw_claim.text,
                status=_status(raw_claim.id, statuses),
            )

    aggregates: dict[str, Aggregate] = {}
    degree: dict[str, int] = {}
    for key, members in grouped.items():
        confirmed = [m for m in members if m.status == "confirmed"]
        left, edge_type, right = aggregate_shape[key]
        traversable = bool(confirmed)
        aggregates[key] = Aggregate(
            key=key,
            source=left,
            type=edge_type,
            target=right,
            weight=len(confirmed),
            mean_strength=(
                sum(m.strength for m in confirmed) / len(confirmed) if confirmed else 0.0
            ),
            assertion_ids=tuple(sorted(m.id for m in members)),
            traversable=traversable,
        )
        if traversable:
            degree[left] = degree.get(left, 0) + 1
            degree[right] = degree.get(right, 0) + 1

    entities = {
        slug: FoldedEntity(
            slug=slug,
            type=entity_types[slug],
            rank=degree.get(slug, 0),
            note_ids=tuple(entity_notes[slug]),
        )
        for slug in sorted(entity_types)
    }

    return GraphTables(
        entities=entities,
        aggregates=aggregates,
        assertions=assertions,
        claims=claims,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_fold.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/graph/ tests/test_fold.py
git commit -m "feat: pure fold from source records to graph tables"
```

---

## Task 10: Cache synchronisation from the vault

**Files:**
- Create: `mindpalace/index/sync.py`
- Test: `tests/test_sync.py`

**Interfaces:**
- Consumes: `mindpalace.graph.fold.fold`, `mindpalace.index.{db, vectors}`, `mindpalace.vault.store.VaultStore`, `mindpalace.embed.Embedder`
- Produces: `SyncReport(notes, captures, entities, aggregates, issues)`, `sync(conn, store, config, embedder, statuses) -> SyncReport`, `has_drift(conn, store) -> bool`, `DOC_KINDS: tuple[str, ...]`

`sync` is a **full** rebuild of Tier 3 from Tier 1 — no incremental path, for the
same reason `fold` has none. At MVP scale a full pass is milliseconds, and it
makes drift self-healing trivially correct.

- [ ] **Step 1: Write the failing test**

`tests/test_sync.py`:

```python
import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db
from mindpalace.index.sync import has_drift, sync
from mindpalace.models import Capture, EntityInstance, Note, RelationshipAssertion
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={},
    )


@pytest.fixture
def vault(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return paths, VaultStore(paths)


@pytest.fixture
def conn(vault):
    paths, _ = vault
    connection = db.connect(paths.graph_db)
    db.create_schema(connection)
    return connection


def add_note(store, note_id, slug, *, relationships=()):
    note = Note(
        id=note_id,
        derived_from=f"c_{note_id[2:]}",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body=f"Analysis about {slug} and data exhaustion.",
        entities=(EntityInstance(slug, "concept", "A concept."),),
        relationship_assertions=tuple(relationships),
    )
    store.write_note(note, slug)
    return note


def test_sync_populates_graph_tables(conn, vault, config):
    _, store = vault
    add_note(
        store,
        "n_01",
        "scaling-laws",
        relationships=(
            RelationshipAssertion(
                "x_1", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
            ),
        ),
    )
    report = sync(conn, store, config, StubEmbedder(), {"x_1": "confirm"})
    assert report.notes == 1
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
    assert conn.execute("SELECT traversable FROM aggregates").fetchone()[0] == 1


def test_sync_indexes_documents_for_fts(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    hits = list(
        conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("exhaustion",))
    )
    assert ("n_01",) in [tuple(row) for row in hits]


def test_sync_stores_vectors_for_every_indexed_doc(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    vecs = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert docs == vecs > 0


def test_sync_is_idempotent(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    first = sync(conn, store, config, StubEmbedder(), {})
    second = sync(conn, store, config, StubEmbedder(), {})
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == first.captures + 1


def test_malformed_note_becomes_an_issue_without_breaking_the_vault(
    conn, vault, config
):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    (paths.notes / "broken.md").write_text("---\nnot: [closed\n")

    report = sync(conn, store, config, StubEmbedder(), {})

    assert any("broken.md" in issue[0] for issue in report.issues)
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] >= 1
    stored = conn.execute("SELECT COUNT(*) FROM vault_issues").fetchone()[0]
    assert stored == len(report.issues)


def test_malformed_file_stays_searchable(conn, vault, config):
    """Spec §10: a file we cannot parse is still indexed for FTS, not dropped."""
    paths, store = vault
    (paths.notes / "broken.md").write_text(
        "---\nnot: [closed\n\nlithium niobate photonics\n"
    )
    sync(conn, store, config, StubEmbedder(), {})

    hits = [
        row[0]
        for row in conn.execute("SELECT doc_id FROM docs WHERE docs MATCH ?", ("niobate",))
    ]
    assert hits == ["notes/broken.md"]


def test_duplicate_assertion_ids_are_reported(conn, vault, config):
    _, store = vault
    shared = RelationshipAssertion(
        "x_dup", "scaling-laws", "data-exhaustion", "contradicts", 8, "because."
    )
    add_note(store, "n_01", "scaling-laws", relationships=(shared,))
    add_note(store, "n_02", "chinchilla", relationships=(shared,))

    report = sync(conn, store, config, StubEmbedder(), {})
    assert any(issue[1] == "duplicate_assertion_id" for issue in report.issues)


def test_sync_records_the_embedder_in_cache_meta(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    embedder = StubEmbedder()
    sync(conn, store, config, embedder, {})
    assert db.read_meta(conn) == (embedder.model_id, embedder.dim)


def test_sync_maintains_index_md(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})

    text = paths.index_md.read_text()
    assert "n_01" in text
    assert "[[scaling-laws]]" in text


def test_has_drift_detects_an_external_edit(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    target = next(paths.notes.glob("*.md"))
    target.write_text(target.read_text() + "\nEdited in Obsidian.\n")
    assert has_drift(conn, store) is True


def test_has_drift_covers_the_decision_log(conn, vault, config):
    """Decisions are Tier 1 — an external edit must not leave the cache stale."""
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    paths.decisions_log.parent.mkdir(parents=True, exist_ok=True)
    paths.decisions_log.write_text("")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    paths.decisions_log.write_text(
        '{"op":"op_1","ts":"2026-08-01T00:00:00Z","assertion":"x_1",'
        '"action":"confirm","via":"resolve_assertion","reason":null}\n'
    )
    assert has_drift(conn, store) is True


def test_has_drift_covers_the_config(conn, vault, config):
    paths, store = vault
    add_note(store, "n_01", "scaling-laws")
    paths.mindpalace_md.write_text("---\nschema_version: 1\n---\n")
    sync(conn, store, config, StubEmbedder(), {})
    assert has_drift(conn, store) is False

    paths.mindpalace_md.write_text("---\nschema_version: 1\n---\n\nchanged\n")
    assert has_drift(conn, store) is True


def test_has_drift_detects_a_new_file(conn, vault, config):
    _, store = vault
    add_note(store, "n_01", "scaling-laws")
    sync(conn, store, config, StubEmbedder(), {})
    add_note(store, "n_02", "chinchilla")
    assert has_drift(conn, store) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.index.sync'`

- [ ] **Step 3: Implement synchronisation**

`mindpalace/index/sync.py`:

```python
"""Full rebuild of the Tier 3 cache from Tier 1 sources."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from mindpalace.atomic import atomic_write, content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.graph.fold import GraphTables, fold
from mindpalace.index import db, vectors
from mindpalace.models import Capture, Note, capture_from_markdown, note_from_markdown
from mindpalace.vault.store import VaultStore

DOC_KINDS = ("capture", "note", "entity", "report", "unparsed")


@dataclass(frozen=True)
class SyncReport:
    notes: int
    captures: int
    entities: int
    aggregates: int
    issues: tuple[tuple[str, str, str], ...] = field(default=())


def _relative(store: VaultStore, path: Path) -> str:
    return str(path.relative_to(store.paths.root))


def _iter_source_files(store: VaultStore) -> Iterator[Path]:
    """Every file the derived cache depends on.

    Includes the decision log and MINDPALACE.md. Both are inputs — decisions
    determine assertion status, config determines how the fold groups edges — so
    omitting them means an external edit to either leaves the cache silently
    stale with nothing able to notice.
    """
    yield from sorted(store.paths.captures.glob("*.md"))
    yield from sorted(store.paths.notes.glob("*.md"))
    for dependency in (store.paths.decisions_log, store.paths.mindpalace_md):
        if dependency.exists():
            yield dependency


def _load_sources(
    store: VaultStore,
) -> tuple[
    list[Capture], list[Note], list[tuple[str, str, str]], list[tuple[str, str]]
]:
    """Parse every source file, degrading rather than dropping on failure."""
    captures: list[Capture] = []
    notes: list[Note] = []
    issues: list[tuple[str, str, str]] = []
    degraded: list[tuple[str, str]] = []

    for path in sorted(store.paths.captures.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            captures.append(capture_from_markdown(raw))
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_capture", str(exc)))
            degraded.append((_relative(store, path), raw))

    for path in sorted(store.paths.notes.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            notes.append(note_from_markdown(raw))
        except Exception as exc:
            issues.append((_relative(store, path), "malformed_note", str(exc)))
            degraded.append((_relative(store, path), raw))

    # The fold keys assertions by id, so a collision would silently drop one.
    seen: dict[str, str] = {}
    for note in notes:
        for assertion in (*note.relationship_assertions, *note.claim_assertions):
            if assertion.id in seen:
                issues.append(
                    (
                        f"notes/{note.id}",
                        "duplicate_assertion_id",
                        f"{assertion.id} is also asserted in note {seen[assertion.id]}",
                    )
                )
            seen[assertion.id] = note.id

    return captures, notes, issues, degraded


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def _render_index(
    captures: list[Capture], notes: list[Note], tables: GraphTables
) -> str:
    """The human- and Obsidian-readable catalog promised by spec §4.5."""
    lines = [
        "# Index",
        "",
        f"{len(captures)} captures · {len(notes)} notes · "
        f"{len(tables.entities)} entities",
        "",
        "## Notes",
        "",
    ]
    for note in sorted(notes, key=lambda item: (item.created, item.id)):
        lines.append(f"- `{note.id}` {note.created[:10]} — {_first_line(note.body)}")
    lines += ["", "## Entities", ""]
    for entity in sorted(
        tables.entities.values(), key=lambda item: (-item.rank, item.slug)
    ):
        lines.append(f"- [[{entity.slug}]] — {entity.type}, rank {entity.rank}")
    return "\n".join(lines) + "\n"


def sync(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    embedder: Embedder,
    statuses: dict[str, str],
) -> SyncReport:
    captures, notes, issues, degraded = _load_sources(store)
    tables = fold(notes, statuses, config)

    documents: list[tuple[str, str, str, str]] = []
    for capture in captures:
        documents.append((capture.id, "capture", _first_line(capture.text), capture.text))
    for note in notes:
        # Assertion descriptions ride along in the note's searchable text rather
        # than becoming their own documents: they must be findable (spec §8.1)
        # but an `x_` id is not something `read` can return.
        assertion_text = " ".join(
            a.description for a in note.relationship_assertions
        )
        claim_text = " ".join(c.text for c in note.claim_assertions)
        searchable = "\n".join(filter(None, [note.body, assertion_text, claim_text]))
        documents.append((note.id, "note", _first_line(note.body), searchable))
    for page in store.iter_entity_pages():
        documents.append((f"e_{page.slug}", "entity", page.slug, page.description))
    for report in store.iter_reports():
        documents.append(
            (report.lineage_id, "report", report.title, f"{report.summary}\n{report.findings}")
        )
    # A file we cannot parse is still findable — spec §10 requires it stay
    # searchable rather than vanishing. Keyed by path, since it has no usable id.
    for relative_path, raw in degraded:
        documents.append((relative_path, "unparsed", relative_path, raw))

    # Embed before opening the transaction. The model call is the slow part and
    # must not hold a write transaction open across it.
    matrix = embedder.embed([text for _, _, _, text in documents]) if documents else []

    with conn:
        for table in (
            "files",
            "entities",
            "entity_sources",
            "assertions",
            "aggregates",
            "aggregate_members",
            "claims",
            "docs",
            "vault_issues",
        ):
            conn.execute(f"DELETE FROM {table}")
        vectors.clear(conn)
        db.write_meta(conn, embedder.model_id, embedder.dim)

        for path in _iter_source_files(store):
            conn.execute(
                "INSERT INTO files (path, hash) VALUES (?, ?)",
                (_relative(store, path), content_hash(path.read_text(encoding="utf-8"))),
            )

        for entity in tables.entities.values():
            conn.execute(
                "INSERT INTO entities (slug, type, rank) VALUES (?, ?, ?)",
                (entity.slug, entity.type, entity.rank),
            )
            conn.executemany(
                "INSERT INTO entity_sources (slug, note_id) VALUES (?, ?)",
                [(entity.slug, note_id) for note_id in entity.note_ids],
            )

        for assertion in tables.assertions.values():
            conn.execute(
                "INSERT INTO assertions (id, kind, note_id, source, target, type, "
                "strength, description, status) VALUES (?, 'relationship', ?, ?, ?, ?, ?, ?, ?)",
                (
                    assertion.id,
                    assertion.note_id,
                    assertion.source,
                    assertion.target,
                    assertion.type,
                    assertion.strength,
                    assertion.description,
                    assertion.status,
                ),
            )

        for claim in tables.claims.values():
            conn.execute(
                "INSERT INTO claims (id, note_id, subject, text, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (claim.id, claim.note_id, claim.subject, claim.text, claim.status),
            )

        for aggregate in tables.aggregates.values():
            conn.execute(
                "INSERT INTO aggregates (key, source, type, target, weight, "
                "mean_strength, traversable) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    aggregate.key,
                    aggregate.source,
                    aggregate.type,
                    aggregate.target,
                    aggregate.weight,
                    aggregate.mean_strength,
                    int(aggregate.traversable),
                ),
            )
            conn.executemany(
                "INSERT INTO aggregate_members (key, assertion_id) VALUES (?, ?)",
                [(aggregate.key, aid) for aid in aggregate.assertion_ids],
            )

        conn.executemany(
            "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
            documents,
        )
        conn.executemany(
            "INSERT INTO vault_issues (path, kind, detail) VALUES (?, ?, ?)", issues
        )

        for (doc_id, kind, _, _), vector in zip(documents, matrix, strict=True):
            vectors.store(conn, doc_id, kind, embedder.model_id, vector)

    atomic_write(store.paths.index_md, _render_index(captures, notes, tables))

    return SyncReport(
        notes=len(notes),
        captures=len(captures),
        entities=len(tables.entities),
        aggregates=len(tables.aggregates),
        issues=tuple(issues),
    )


def has_drift(conn: sqlite3.Connection, store: VaultStore) -> bool:
    recorded = {
        row["path"]: row["hash"] for row in conn.execute("SELECT path, hash FROM files")
    }
    seen: set[str] = set()
    for path in _iter_source_files(store):
        relative = _relative(store, path)
        seen.add(relative)
        current = content_hash(path.read_text(encoding="utf-8"))
        if recorded.get(relative) != current:
            return True
    return seen != recorded.keys()
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_sync.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/index/sync.py tests/test_sync.py
git commit -m "feat: full cache synchronisation with drift detection"
```

---

## Task 11: Local search — RRF ordering with an evidence gate

**Files:**
- Create: `mindpalace/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `mindpalace.index.vectors.search`, `mindpalace.config.Config`, `mindpalace.embed.Embedder`
- Produces: `Hit(id, kind, title, snippet, bm25, cosine, score)`, `rrf(rank_lists, k=60) -> list[tuple[str, float]]`, `passes_evidence_gate(hits, bm25_floor, cosine_floor) -> bool`, `local_search(conn, embedder, query, config, k=8, expand_graph=False) -> dict`, `LOCAL_KINDS: tuple[str, ...]`, `RRF_K: int`

**The gate must not use the RRF score.** RRF sums `1/(k + rank)`, which encodes
ordering only — its magnitude shifts with `k`, list length, and corpus size, so a
fixed threshold over it means something different every week. Relevance is judged
on raw BM25 and raw cosine, which are comparable across corpus sizes.

- [ ] **Step 1: Write the failing test**

`tests/test_retrieve.py`:

```python
import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db, vectors
from mindpalace.retrieve import Hit, local_search, passes_evidence_gate, rrf


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={"abstain": "Nothing in the vault is relevant to this query."},
    )


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "cache.db")
    db.create_schema(connection)
    return connection


def index_doc(conn, embedder, doc_id, kind, title, text):
    conn.execute(
        "INSERT INTO docs (doc_id, kind, title, text) VALUES (?, ?, ?, ?)",
        (doc_id, kind, title, text),
    )
    conn.commit()
    vectors.store(conn, doc_id, kind, embedder.model_id, embedder.embed([text])[0])


def hit(doc_id, *, bm25=0.0, cosine=0.0):
    return Hit(
        id=doc_id, kind="note", title="t", snippet="s", bm25=bm25, cosine=cosine, score=1.0
    )


def test_rrf_prefers_documents_appearing_in_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "b", "a"]])
    assert fused[0][0] == "b"


def test_rrf_respects_rank_order_within_a_single_list():
    assert [doc for doc, _ in rrf([["a", "b", "c"]])] == ["a", "b", "c"]


def test_rrf_handles_an_empty_list():
    assert rrf([[], ["a"]])[0][0] == "a"


def test_gate_rejects_when_both_signals_are_weak():
    assert passes_evidence_gate([hit("n_01", bm25=0.4, cosine=0.1)], 2.0, 0.35) is False


def test_gate_accepts_on_lexical_evidence_alone():
    assert passes_evidence_gate([hit("n_01", bm25=5.0, cosine=0.05)], 2.0, 0.35) is True


def test_gate_accepts_on_semantic_evidence_alone():
    assert passes_evidence_gate([hit("n_01", bm25=0.0, cosine=0.9)], 2.0, 0.35) is True


def test_gate_rejects_an_empty_result_set():
    assert passes_evidence_gate([], 2.0, 0.35) is False


def test_local_search_abstains_with_an_explicit_instruction(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Bread", "sourdough starter hydration")
    result = local_search(conn, embedder, "quantum chromodynamics", config)
    assert result["hits"] == []
    assert "rather than answering from your own knowledge" in result["note"]


def test_local_search_returns_lexical_matches(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Scaling", "the plateau is data exhaustion")
    index_doc(conn, embedder, "n_02", "note", "Bread", "sourdough starter hydration")
    result = local_search(conn, embedder, "data exhaustion", config)
    assert [h["id"] for h in result["hits"]][0] == "n_01"
    assert result["hits"][0]["bm25"] > 0


def test_local_search_reports_raw_signals_for_tuning(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "n_01", "note", "Scaling", "the plateau is data exhaustion")
    result = local_search(conn, embedder, "data exhaustion", config)
    assert "max_bm25" in result["signals"]
    assert "max_cosine" in result["signals"]


def test_expand_graph_follows_only_traversable_aggregates(conn, config):
    embedder = StubEmbedder()
    index_doc(conn, embedder, "e_scaling-laws", "entity", "scaling-laws", "scaling laws")
    conn.executemany(
        "INSERT INTO aggregates (key, source, type, target, weight, mean_strength, "
        "traversable) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("r:a|contradicts|scaling-laws", "a", "contradicts", "scaling-laws", 1, 8.0, 1),
            ("r:b|contradicts|scaling-laws", "b", "contradicts", "scaling-laws", 0, 0.0, 0),
        ],
    )
    conn.commit()
    result = local_search(conn, embedder, "scaling laws", config, expand_graph=True)
    neighbours = {n["slug"] for n in result["neighbours"]}
    assert neighbours == {"a"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.retrieve'`

- [ ] **Step 3: Implement local search**

`mindpalace/retrieve.py`:

```python
"""Retrieval. RRF orders results; a separate evidence gate decides relevance."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.index import vectors

# LOCAL_KINDS includes "unparsed" so a file we could not parse is still
# reachable by search rather than silently absent from the vault.
LOCAL_KINDS = ("capture", "note", "entity", "unparsed")
RRF_K = 60
CANDIDATES = 40
TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Hit:
    id: str
    kind: str
    title: str
    snippet: str
    bm25: float
    cosine: float
    score: float


def rrf(rank_lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for position, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def passes_evidence_gate(
    hits: Sequence[Hit], bm25_floor: float, cosine_floor: float
) -> bool:
    """Relevance is judged on raw signals, never on the fused ordering score."""
    return any(hit.bm25 >= bm25_floor or hit.cosine >= cosine_floor for hit in hits)


def _fts_query(text: str) -> str:
    terms = TOKEN.findall(text)
    return " OR ".join(f'"{term}"' for term in terms)


def _fts_candidates(
    conn: sqlite3.Connection, query: str, kinds: Sequence[str], limit: int
) -> dict[str, dict]:
    expression = _fts_query(query)
    if not expression:
        return {}
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT doc_id, kind, title, -bm25(docs) AS score, "
        f"snippet(docs, 3, '', '', '…', 12) AS snippet "
        f"FROM docs WHERE docs MATCH ? AND kind IN ({placeholders}) "
        f"ORDER BY bm25(docs) LIMIT ?",
        (expression, *kinds, limit),
    ).fetchall()
    return {
        row["doc_id"]: {
            "kind": row["kind"],
            "title": row["title"],
            "snippet": row["snippet"],
            "bm25": max(0.0, float(row["score"])),
        }
        for row in rows
    }


def _doc_meta(conn: sqlite3.Connection, doc_id: str) -> dict:
    row = conn.execute(
        "SELECT kind, title, substr(text, 1, 200) AS snippet FROM docs WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return {"kind": "", "title": "", "snippet": ""}
    return {"kind": row["kind"], "title": row["title"], "snippet": row["snippet"]}


def local_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    config: Config,
    k: int = 8,
    expand_graph: bool = False,
) -> dict:
    query_vector = embedder.embed([query])[0]
    vector_hits = vectors.search(
        conn, query_vector, embedder.model_id, LOCAL_KINDS, CANDIDATES
    )
    cosines = {doc_id: score for doc_id, score in vector_hits}
    lexical = _fts_candidates(conn, query, LOCAL_KINDS, CANDIDATES)

    fused = rrf([[doc_id for doc_id, _ in vector_hits], list(lexical)])

    hits: list[Hit] = []
    for doc_id, score in fused[:k]:
        meta = lexical.get(doc_id) or _doc_meta(conn, doc_id)
        hits.append(
            Hit(
                id=doc_id,
                kind=meta["kind"],
                title=meta["title"],
                snippet=meta["snippet"],
                bm25=float(meta.get("bm25", 0.0)),
                cosine=float(cosines.get(doc_id, 0.0)),
                score=score,
            )
        )

    signals = {
        "max_bm25": max((h.bm25 for h in hits), default=0.0),
        "max_cosine": max((h.cosine for h in hits), default=0.0),
        "bm25_floor": config.thresholds.abstain_bm25_floor,
        "cosine_floor": config.thresholds.abstain_cosine_floor,
    }

    if not passes_evidence_gate(
        hits,
        config.thresholds.abstain_bm25_floor,
        config.thresholds.abstain_cosine_floor,
    ):
        return {
            "hits": [],
            "neighbours": [],
            "signals": signals,
            "note": config.templates.get(
                "abstain",
                "Nothing in the vault is relevant to this query. Say so rather "
                "than answering from your own knowledge.",
            ),
        }

    neighbours = _expand(conn, hits) if expand_graph else []
    return {
        "hits": [asdict(hit) for hit in hits],
        "neighbours": neighbours,
        "signals": signals,
    }


def _expand(conn: sqlite3.Connection, hits: Sequence[Hit]) -> list[dict]:
    slugs = [hit.id.removeprefix("e_") for hit in hits if hit.kind == "entity"]
    if not slugs:
        return []
    placeholders = ",".join("?" for _ in slugs)
    rows = conn.execute(
        f"SELECT source, target, type, weight FROM aggregates "
        f"WHERE traversable = 1 AND (source IN ({placeholders}) "
        f"OR target IN ({placeholders}))",
        (*slugs, *slugs),
    ).fetchall()

    seen: set[str] = set()
    neighbours: list[dict] = []
    for row in rows:
        for slug in (row["source"], row["target"]):
            if slug in slugs or slug in seen:
                continue
            seen.add(slug)
            neighbours.append(
                {"slug": slug, "via": row["type"], "weight": row["weight"]}
            )
    return neighbours
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/retrieve.py tests/test_retrieve.py
git commit -m "feat: local search with RRF ordering and raw-signal evidence gate"
```

---

## Task 12: Hierarchical clustering with stable lineage

**Files:**
- Create: `mindpalace/cluster.py`
- Test: `tests/test_cluster.py`
- Modify: `pyproject.toml` (add `graspologic` and `networkx`)

**Interfaces:**
- Consumes: `mindpalace.graph.fold.Aggregate`, `mindpalace.config.Config`, `mindpalace.ids.new_id`
- Produces: `Community(lineage_id, level, members, parent)`, `jaccard(a, b) -> float`, `match_lineages(fresh, previous, threshold) -> list[Community]`, `should_cluster(entity_count, threshold) -> bool`, `partition(aggregates, config, seed=42) -> list[Community]`, `ClusterBelowThreshold`

- [ ] **Step 1: Write the failing test**

`tests/test_cluster.py`:

```python
import pytest

from mindpalace.cluster import (
    Community,
    jaccard,
    match_lineages,
    partition,
    should_cluster,
)
from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.graph.fold import Aggregate


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0),
            "mentions": EdgeType("mentions", directed=True, cluster_weight=0.0),
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )


def aggregate(source, target, edge_type="contradicts", traversable=True, weight=1):
    return Aggregate(
        key=f"r:{source}|{edge_type}|{target}",
        source=source,
        type=edge_type,
        target=target,
        weight=weight,
        mean_strength=5.0,
        assertion_ids=("x_1",),
        traversable=traversable,
    )


def test_should_cluster_respects_the_activation_threshold():
    assert should_cluster(149, 150) is False
    assert should_cluster(150, 150) is True


def test_jaccard_of_identical_sets_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_of_empty_sets_is_zero():
    assert jaccard(set(), set()) == 0.0


def test_partition_is_deterministic_under_a_fixed_seed(config):
    aggregates = {
        a.key: a
        for a in [
            aggregate("a", "b"),
            aggregate("b", "c"),
            aggregate("x", "y"),
            aggregate("y", "z"),
        ]
    }
    first = partition(aggregates, config, seed=42)
    second = partition(aggregates, config, seed=42)
    assert [sorted(c.members) for c in first] == [sorted(c.members) for c in second]


def test_partition_ignores_untraversable_aggregates(config):
    aggregates = {
        a.key: a
        for a in [aggregate("a", "b"), aggregate("ghost", "phantom", traversable=False)]
    }
    members = {slug for community in partition(aggregates, config) for slug in community.members}
    assert "ghost" not in members
    assert "a" in members


def test_partition_ignores_zero_weight_edge_types(config):
    """A cluster_weight of 0.0 removes a type from clustering entirely."""
    aggregates = {
        a.key: a for a in [aggregate("a", "b", edge_type="mentions")]
    }
    assert partition(aggregates, config) == []


def test_match_lineages_preserves_id_when_membership_overlaps(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b", "c"}), parent=None)]
    fresh = [Community(lineage_id="g_new", level=0, members=frozenset({"a", "b", "d"}), parent=None)]
    [matched] = match_lineages(fresh, previous, 0.5)
    assert matched.lineage_id == "g_old"
    assert matched.members == frozenset({"a", "b", "d"})


def test_match_lineages_mints_a_new_id_when_overlap_is_too_low(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b", "c"}), parent=None)]
    fresh = [Community(lineage_id="g_new", level=0, members=frozenset({"x", "y", "z"}), parent=None)]
    [matched] = match_lineages(fresh, previous, 0.5)
    assert matched.lineage_id == "g_new"


def test_each_previous_lineage_is_claimed_at_most_once(config):
    previous = [Community(lineage_id="g_old", level=0, members=frozenset({"a", "b"}), parent=None)]
    fresh = [
        Community(lineage_id="g_1", level=0, members=frozenset({"a", "b"}), parent=None),
        Community(lineage_id="g_2", level=0, members=frozenset({"a", "b"}), parent=None),
    ]
    ids = {community.lineage_id for community in match_lineages(fresh, previous, 0.5)}
    assert "g_old" in ids
    assert len(ids) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cluster.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.cluster'`

- [ ] **Step 3: Add the clustering dependency**

Edit `pyproject.toml` — add `graspologic` and `networkx` to `dependencies`
(graspologic pulls networkx in transitively, but this module imports it directly,
so declaring it is correct):

```toml
dependencies = [
    "mcp>=1.2.0",
    "pyyaml>=6.0",
    "python-ulid>=2.7",
    "numpy>=1.26",
    "fastembed>=0.4",
    "graspologic>=3.4",
    "networkx>=3.0",
]
```

Run: `uv sync`

- [ ] **Step 4: Implement clustering**

`mindpalace/cluster.py`:

```python
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
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_cluster.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add mindpalace/cluster.py pyproject.toml tests/test_cluster.py
git commit -m "feat: seeded hierarchical Leiden with stable community lineage"
```

---

## Task 13: Citations and global search

**Files:**
- Create: `mindpalace/citations.py`
- Modify: `mindpalace/retrieve.py` (append `global_search`)
- Test: `tests/test_citations.py`, `tests/test_global_search.py`

**Interfaces:**
- Consumes: `mindpalace.vault.store.VaultStore`, `mindpalace.retrieve.rrf`
- Produces: `citations.extract_ids(text) -> list[str]`, `citations.unresolvable(cites, resolver) -> list[str]`, `citations.CitationError`, `citations.CITATION_PATTERN`; `retrieve.global_search(conn, store, embedder, query, config) -> dict`

- [ ] **Step 1: Write the failing citations test**

`tests/test_citations.py`:

```python
from mindpalace.citations import extract_ids, unresolvable


def test_extract_ids_from_a_graphrag_style_citation():
    text = "The plateau is contested [Data: Entities (e_scaling-laws); Assertions (x_01ab)]."
    assert extract_ids(text) == ["e_scaling-laws", "x_01ab"]


def test_extract_ids_across_multiple_citations():
    text = "One [Data: Entities (e_a)]. Two [Data: Assertions (x_b, k_c)]."
    assert extract_ids(text) == ["e_a", "x_b", "k_c"]


def test_extract_ids_returns_empty_for_ungrounded_prose():
    assert extract_ids("A confident claim with no support.") == []


def test_extract_ids_ignores_the_more_marker():
    text = "[Data: Entities (e_a, e_b, +more)]"
    assert extract_ids(text) == ["e_a", "e_b"]


def test_unresolvable_reports_only_missing_ids():
    known = {"e_a", "x_b"}
    assert unresolvable(["e_a", "x_b", "e_ghost"], known.__contains__) == ["e_ghost"]


def test_unresolvable_is_empty_when_everything_resolves():
    assert unresolvable(["e_a"], {"e_a"}.__contains__) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_citations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.citations'`

- [ ] **Step 3: Implement citations**

`mindpalace/citations.py`:

```python
"""Extraction and validation of inline data citations.

Prose that cannot be checked is prose that cannot be repaired, so every
generated artifact carries both inline citations and a structural `cites` list.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

CITATION_PATTERN = re.compile(r"\[Data:([^\]]*)\]")
ID_PATTERN = re.compile(r"\b((?:e|x|k|n|c|g)_[A-Za-z0-9\-]+)\b")


class CitationError(ValueError):
    """Raised when a generated artifact cites something that does not exist."""


def extract_ids(text: str) -> list[str]:
    found: list[str] = []
    for block in CITATION_PATTERN.findall(text):
        for identifier in ID_PATTERN.findall(block):
            if identifier not in found:
                found.append(identifier)
    return found


def unresolvable(
    cites: Iterable[str], resolver: Callable[[str], bool]
) -> list[str]:
    return [identifier for identifier in cites if not resolver(identifier)]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_citations.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing global search test**

`tests/test_global_search.py`:

```python
import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db
from mindpalace.models import CommunityReport
from mindpalace.retrieve import global_search
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={"report_next": "Ground every finding."},
    )


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


@pytest.fixture
def conn(store):
    connection = db.connect(store.paths.graph_db)
    db.create_schema(connection)
    return connection


def add_community(conn, lineage_id, members, level=0):
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, ?, ?, ?)",
        (lineage_id, level, None, ",".join(sorted(members))),
    )
    conn.commit()


def add_entities(conn, count):
    for index in range(count):
        conn.execute(
            "INSERT INTO entities (slug, type, rank) VALUES (?, 'concept', 1)",
            (f"e{index}",),
        )
    conn.commit()


def test_unavailable_below_threshold_explains_rather_than_errors(conn, store, config):
    add_entities(conn, 3)
    result = global_search(conn, store, StubEmbedder(), "themes?", config)
    assert result["available"] is False
    assert result["entity_count"] == 3
    assert result["threshold"] == 150
    assert "local_search" in result["note"]


def test_available_once_communities_exist(conn, store, config):
    add_entities(conn, 3)
    add_community(conn, "g_01", {"a", "b"})
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="A cluster about scaling limits.",
            rank=7.0,
        )
    )
    result = global_search(conn, store, StubEmbedder(), "scaling", config)
    assert result["available"] is True
    assert result["communities"][0]["lineage_id"] == "g_01"
    assert result["communities"][0]["report"] == "present"


def test_missing_report_is_flagged_and_keeps_its_members(conn, store, config):
    add_community(conn, "g_02", {"x", "y"})
    result = global_search(conn, store, StubEmbedder(), "anything", config)
    [community] = result["communities"]
    assert community["report"] == "missing"
    assert community["members"] == ["x", "y"]


def test_stale_report_is_flagged_but_still_returned(conn, store, config):
    add_community(conn, "g_03", {"a"})
    store.write_report(
        CommunityReport(
            lineage_id="g_03",
            level=0,
            title="Old News",
            summary="Written before the graph moved.",
            rank=4.0,
            stale=True,
        )
    )
    [community] = global_search(conn, store, StubEmbedder(), "news", config)["communities"]
    assert community["report"] == "stale"
    assert community["summary"] == "Written before the graph moved."


def test_higher_impact_rank_breaks_ties(conn, store, config):
    add_community(conn, "g_low", {"a"})
    add_community(conn, "g_high", {"b"})
    for lineage_id, rank in (("g_low", 2.0), ("g_high", 9.0)):
        store.write_report(
            CommunityReport(
                lineage_id=lineage_id,
                level=0,
                title=f"Report {lineage_id}",
                summary="Identical text for both communities.",
                rank=rank,
            )
        )
    result = global_search(conn, store, StubEmbedder(), "identical text", config)
    assert result["communities"][0]["lineage_id"] == "g_high"


def test_instructions_come_from_the_config_template(conn, store, config):
    add_community(conn, "g_04", {"a"})
    result = global_search(conn, store, StubEmbedder(), "q", config)
    assert result["instructions"] == "Ground every finding."
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_global_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'global_search'`

- [ ] **Step 7: Append `global_search` to `mindpalace/retrieve.py`**

```python
REPORT_KIND = ("report",)
RANK_TIEBREAK_WEIGHT = 0.001


def global_search(
    conn: sqlite3.Connection,
    store,
    embedder: Embedder,
    query: str,
    config: Config,
) -> dict:
    """Rank community reports for the assistant to answer from.

    At personal-vault scale ten to twenty reports fit in one context window, so
    the paper's distributed map-reduce is unnecessary; the assistant answers
    directly. Communities lacking a usable report are still returned with their
    members so no region of the graph silently vanishes.
    """
    communities = conn.execute(
        "SELECT lineage_id, level, members FROM communities ORDER BY level, lineage_id"
    ).fetchall()
    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    threshold = config.thresholds.cluster_activation_entities

    if not communities:
        return {
            "available": False,
            "entity_count": entity_count,
            "threshold": threshold,
            "communities": [],
            "note": (
                f"No communities exist yet ({entity_count} entities; clustering "
                f"activates at {threshold}). Use local_search for this query."
            ),
        }

    reports = {report.lineage_id: report for report in store.iter_reports()}

    query_vector = embedder.embed([query])[0]
    relevance = dict(
        vectors.search(conn, query_vector, embedder.model_id, REPORT_KIND, CANDIDATES)
    )
    lexical = _fts_candidates(conn, query, REPORT_KIND, CANDIDATES)
    ordering = {
        doc_id: score
        for doc_id, score in rrf([list(relevance), list(lexical)])
    }

    entries = []
    for row in communities:
        lineage_id = row["lineage_id"]
        report = reports.get(lineage_id)
        if report is None:
            state = "missing"
        elif report.stale:
            state = "stale"
        else:
            state = "present"
        entries.append(
            {
                "lineage_id": lineage_id,
                "level": row["level"],
                "members": row["members"].split(",") if row["members"] else [],
                "report": state,
                "title": report.title if report else None,
                "summary": report.summary if report else None,
                "rank": report.rank if report else 0.0,
                "findings": report.findings if report else [],
                "relevance": ordering.get(lineage_id, 0.0),
            }
        )

    entries.sort(
        key=lambda entry: (
            -(entry["relevance"] + RANK_TIEBREAK_WEIGHT * entry["rank"]),
            entry["lineage_id"],
        )
    )

    return {
        "available": True,
        "entity_count": entity_count,
        "threshold": threshold,
        "communities": entries,
        "instructions": config.templates.get("report_next", ""),
    }
```

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_global_search.py tests/test_retrieve.py -v`
Expected: PASS (17 tests)

- [ ] **Step 9: Commit**

```bash
git add mindpalace/citations.py mindpalace/retrieve.py tests/test_citations.py tests/test_global_search.py
git commit -m "feat: citation validation and global search over community reports"
```

---

## Task 14: Rebuild and staleness marking

**Files:**
- Create: `mindpalace/rebuild.py`
- Test: `tests/test_rebuild.py`

**Interfaces:**
- Consumes: `mindpalace.index.sync.sync`, `mindpalace.graph.fold.fold`, `mindpalace.vault.store.VaultStore`
- Produces: `RebuildReport(synced, pages_written, pages_marked_stale, reports_marked_stale)`, `entity_input_hash(entity, tables, notes_by_id) -> str`, `community_input_hash(members, tables) -> str`, `mark_stale_reports(conn, store, tables) -> int`, `rebuild(conn, store, config, embedder, statuses, scope="all") -> RebuildReport`

Both hashes cover **evidence**, not identity. Hashing ids alone would let a
rewritten note leave its entity page reading `stale: false` forever.

**Rebuild never deletes Tier 2 and never regenerates prose.** It recomputes
input hashes, marks prose stale where inputs moved, and regenerates only the
machine-owned `related` block.

- [ ] **Step 1: Write the failing test**

`tests/test_rebuild.py`:

```python
import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.embed import StubEmbedder
from mindpalace.index import db
from mindpalace.models import CommunityReport, EntityInstance, EntityPage, Note, RelationshipAssertion
from mindpalace.rebuild import rebuild
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0)
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub", "dim": 64},
        templates={},
    )


@pytest.fixture
def store(tmp_path):
    paths = VaultPaths(tmp_path)
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return VaultStore(paths)


@pytest.fixture
def conn(store):
    connection = db.connect(store.paths.graph_db)
    db.create_schema(connection)
    return connection


def add_note(store, note_id="n_01", target="data-exhaustion"):
    note = Note(
        id=note_id,
        derived_from="c_01",
        created="2026-08-01T00:00:00Z",
        author="llm",
        body="Analysis.",
        entities=(EntityInstance("scaling-laws", "concept", "A concept."),),
        relationship_assertions=(
            RelationshipAssertion(
                f"x_{note_id}", "scaling-laws", target, "contradicts", 8, "because."
            ),
        ),
    )
    store.write_note(note, f"note-{note_id}")
    return note


def test_rebuild_creates_missing_entity_pages_as_stale(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page is not None
    assert page.stale is True
    assert page.description == ""


def test_rebuild_preserves_prose_and_user_overrides(conn, store, config):
    add_note(store)
    store.write_entity_page(
        EntityPage(
            slug="scaling-laws",
            type="concept",
            description="Hand-written prose that must survive.",
            input_hash="sha256:stale",
            stale=False,
            user={"aliases": ["scaling law"], "type": "theme"},
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page.description == "Hand-written prose that must survive."
    assert page.user == {"aliases": ["scaling law"], "type": "theme"}


def test_rebuild_marks_a_page_stale_when_its_inputs_move(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    store.write_entity_page(
        EntityPage(
            slug=page.slug,
            type=page.type,
            description="Now written.",
            generated_from=page.generated_from,
            input_hash=page.input_hash,
            stale=False,
        )
    )
    add_note(store, note_id="n_02", target="chinchilla")

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert store.read_entity_page("scaling-laws").stale is True
    assert report.pages_marked_stale >= 1


def test_editing_a_note_body_marks_its_entity_page_stale(conn, store, config):
    """The regression that identity-hashing missed: the note id never changes,
    so a hash over ids alone would leave the page reading fresh forever."""
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.description = "Written from the original wording."
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(
        note_path.read_text().replace("Analysis.", "Entirely different wording.")
    )

    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})

    assert store.read_entity_page("scaling-laws").stale is True
    assert report.pages_marked_stale == 1


def test_editing_an_assertion_rationale_marks_the_page_stale(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    page.stale = False
    store.write_entity_page(page)

    note_path = next(store.paths.notes.glob("*.md"))
    note_path.write_text(note_path.read_text().replace("because.", "for a new reason."))

    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_entity_page("scaling-laws").stale is True


def test_rebuild_regenerates_the_related_block(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    page = store.read_entity_page("scaling-laws")
    assert page.related == ["contradicts [[data-exhaustion]]"]


def test_related_block_excludes_untraversable_aggregates(conn, store, config):
    add_note(store)
    rebuild(conn, store, config, StubEmbedder(), {})  # nothing confirmed
    assert store.read_entity_page("scaling-laws").related == []


def test_rebuild_never_deletes_community_reports(conn, store, config):
    add_note(store)
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="Prose that no rebuild can recreate.",
            rank=6.0,
        )
    )
    rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    survivor = store.read_report("g_01")
    assert survivor is not None
    assert survivor.summary == "Prose that no rebuild can recreate."


def test_rebuild_marks_a_report_stale_when_membership_changes(conn, store, config):
    add_note(store)
    conn.execute(
        "INSERT INTO communities (lineage_id, level, parent, members) VALUES (?, 0, NULL, ?)",
        ("g_01", "scaling-laws,data-exhaustion,chinchilla"),
    )
    conn.commit()
    store.write_report(
        CommunityReport(
            lineage_id="g_01",
            level=0,
            title="Scaling Debate",
            summary="Summary.",
            rank=6.0,
            input_hash="sha256:different",
            stale=False,
        )
    )
    report = rebuild(conn, store, config, StubEmbedder(), {"x_n_01": "confirm"})
    assert store.read_report("g_01").stale is True
    assert report.reports_marked_stale == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_rebuild.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.rebuild'`

- [ ] **Step 3: Implement rebuild**

`mindpalace/rebuild.py`:

```python
"""Regenerate Tier 3 and mark Tier 2 stale. Never deletes or rewrites prose."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from collections.abc import Iterable

from mindpalace.atomic import content_hash
from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.graph.fold import FoldedEntity, GraphTables, fold
from mindpalace.ids import slugify
from mindpalace.index.sync import sync
from mindpalace.models import EntityPage, Note
from mindpalace.vault.store import VaultStore


@dataclass(frozen=True)
class RebuildReport:
    synced: int
    pages_written: int
    pages_marked_stale: int
    reports_marked_stale: int


def entity_input_hash(
    entity: FoldedEntity, tables: GraphTables, notes_by_id: dict[str, Note]
) -> str:
    """Hash the *evidence* a description was written from, not its identity.

    Hashing ids alone (type, note ids, edge shape) would leave a page reading
    `stale: false` after its source note was rewritten in place — the id never
    changes, so the hash never moves, and the prose silently stops describing
    anything real. Everything a writer could have read has to be in here.
    """
    parts = [f"type={entity.type}"]

    for note_id in sorted(entity.note_ids):
        note = notes_by_id.get(note_id)
        if note is None:
            continue
        parts.append(f"note={note_id}:{content_hash(note.body)}")
        for instance in note.entities:
            if slugify(instance.name) == entity.slug:
                parts.append(f"instance={content_hash(instance.description)}")

    for aggregate in sorted(tables.aggregates.values(), key=lambda item: item.key):
        if not aggregate.traversable:
            continue
        if entity.slug not in (aggregate.source, aggregate.target):
            continue
        parts.append(
            f"edge={aggregate.key}:{aggregate.weight}:{aggregate.mean_strength}"
        )
        for assertion_id in aggregate.assertion_ids:
            assertion = tables.assertions.get(assertion_id)
            if assertion is not None:
                parts.append(
                    f"why={assertion_id}:{assertion.status}:"
                    f"{content_hash(assertion.description)}"
                )

    for claim in sorted(tables.claims.values(), key=lambda item: item.id):
        if claim.subject == entity.slug:
            parts.append(f"claim={claim.id}:{claim.status}:{content_hash(claim.text)}")

    return content_hash("\n".join(parts))


def community_input_hash(members: Iterable[str], tables: GraphTables) -> str:
    """Hash the community's evidence, not just its member slugs.

    Membership can hold steady while every description, weight, and claim under
    it changes — a report hashed on slugs alone would never notice.
    """
    membership = set(members)
    parts = []

    for slug in sorted(membership):
        entity = tables.entities.get(slug)
        parts.append(
            f"entity={slug}" if entity is None else f"entity={slug}:{entity.type}:{entity.rank}"
        )

    for aggregate in sorted(tables.aggregates.values(), key=lambda item: item.key):
        if (
            aggregate.traversable
            and aggregate.source in membership
            and aggregate.target in membership
        ):
            parts.append(
                f"edge={aggregate.key}:{aggregate.weight}:{aggregate.mean_strength}"
            )

    for claim in sorted(tables.claims.values(), key=lambda item: item.id):
        if claim.subject in membership and claim.status == "confirmed":
            parts.append(f"claim={claim.id}:{content_hash(claim.text)}")

    return content_hash("\n".join(parts))


def mark_stale_reports(
    conn: sqlite3.Connection, store: VaultStore, tables: GraphTables
) -> int:
    """Flag any report whose community evidence has moved since it was written.

    Shared by `rebuild` and `cluster_tool`: reclustering must not leave an
    obsolete report reading as `present` in global_search until some unrelated
    rebuild happens to run.
    """
    membership = {
        row["lineage_id"]: (row["members"].split(",") if row["members"] else [])
        for row in conn.execute("SELECT lineage_id, members FROM communities")
    }
    marked = 0
    for report in list(store.iter_reports()):
        members = membership.get(report.lineage_id)
        if members is None:
            continue
        expected = community_input_hash(members, tables)
        if report.input_hash != expected and not report.stale:
            report.stale = True
            report.input_hash = expected
            store.write_report(report)
            marked += 1
    return marked


def _related_lines(slug: str, tables: GraphTables) -> list[str]:
    lines = []
    for aggregate in tables.aggregates.values():
        if not aggregate.traversable:
            continue
        if aggregate.source == slug:
            lines.append(f"{aggregate.type} [[{aggregate.target}]]")
        elif aggregate.target == slug:
            lines.append(f"{aggregate.type} [[{aggregate.source}]]")
    return sorted(set(lines))


def rebuild(
    conn: sqlite3.Connection,
    store: VaultStore,
    config: Config,
    embedder: Embedder,
    statuses: dict[str, str],
    scope: str = "all",
) -> RebuildReport:
    synced = 0
    if scope in {"cache", "all"}:
        synced = sync(conn, store, config, embedder, statuses).notes

    notes = list(store.iter_notes())
    notes_by_id = {note.id: note for note in notes}
    tables = fold(notes, statuses, config)

    written = 0
    marked_stale = 0
    if scope in {"related_blocks", "all"}:
        for entity in tables.entities.values():
            expected = entity_input_hash(entity, tables, notes_by_id)
            existing = store.read_entity_page(entity.slug)
            related = _related_lines(entity.slug, tables)

            if existing is None:
                store.write_entity_page(
                    EntityPage(
                        slug=entity.slug,
                        type=entity.type,
                        description="",
                        generated_from=list(entity.note_ids),
                        input_hash=expected,
                        stale=True,
                        related=related,
                    )
                )
                written += 1
                continue

            became_stale = existing.input_hash != expected
            if became_stale and not existing.stale:
                marked_stale += 1

            store.write_entity_page(
                EntityPage(
                    slug=entity.slug,
                    type=existing.user.get("type", entity.type),
                    description=existing.description,
                    generated_from=list(entity.note_ids),
                    input_hash=expected,
                    stale=existing.stale or became_stale,
                    user=existing.user,
                    related=related,
                )
            )
            written += 1

    reports_marked = mark_stale_reports(conn, store, tables)

    # Entity pages are written *after* the first sync, so their docs and vectors
    # would otherwise lag a full cycle behind. Re-sync once they exist.
    if written or reports_marked:
        sync(conn, store, config, embedder, statuses)

    return RebuildReport(
        synced=synced,
        pages_written=written,
        pages_marked_stale=marked_stale,
        reports_marked_stale=reports_marked,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_rebuild.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add mindpalace/rebuild.py tests/test_rebuild.py
git commit -m "feat: cache rebuild with Tier 2 staleness marking"
```

---

## Task 15: Session — locking, crash healing, and the write-path tools

**Files:**
- Create: `mindpalace/session.py`, `mindpalace/tools.py`
- Test: `tests/test_session.py`, `tests/test_tools_write.py`

**Interfaces:**
- Consumes: everything built so far
- Produces: `Session(root, init=False, embedder=None)` with attributes `paths, config, store, conn, embedder, oplog, decisions` and methods `open()`, `close()`, `operation(intent)` (context manager yielding `op_id`), `heal() -> bool`, `statuses() -> dict[str, str]`; `VaultLockedError`; `tools.save_capture(session, text, why=None, source="manual") -> dict`; `tools.write_note(session, derived_from, content, entities=(), relationship_assertions=(), claim_assertions=()) -> dict`; `tools.ToolError`

Every interrupted operation is healed the same way: a full `sync`. That works
because Tier 3 is derivable, so recovery never needs per-operation undo logic.

- [ ] **Step 1: Write the failing session test**

`tests/test_session.py`:

```python
import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session, VaultLockedError


def open_session(root):
    return Session(root, init=True, embedder=StubEmbedder())


def test_session_initialises_a_fresh_vault(tmp_path):
    with open_session(tmp_path) as session:
        assert session.paths.mindpalace_md.exists()
        assert session.config.schema_version == 1


def test_lock_prevents_a_second_writer(tmp_path):
    with open_session(tmp_path):
        with pytest.raises(VaultLockedError, match="already open"):
            open_session(tmp_path).open()


def test_lock_is_released_on_close(tmp_path):
    with open_session(tmp_path):
        pass
    with open_session(tmp_path) as session:
        assert session.paths.lock.exists()


def test_stale_lock_from_a_dead_process_is_reclaimed(tmp_path):
    """flock is released by the kernel when the holder dies, so a leftover pid
    file never blocks a restart."""
    with open_session(tmp_path) as session:
        lock_path = session.paths.lock
    lock_path.write_text("999999")  # pid that cannot be running
    with open_session(tmp_path) as session:
        assert session.paths.lock.read_text() != "999999"


def test_changing_the_embedder_rebuilds_the_cache(tmp_path):
    """A swapped model does not error — vectors filter on model_id, so it would
    silently return nothing. Startup must detect and rebuild."""
    with open_session(tmp_path) as session:
        session.conn.execute(
            "INSERT INTO docs (doc_id, kind, title, text) VALUES "
            "('n_stale', 'note', 't', 'text')"
        )
        session.conn.commit()

    with Session(tmp_path, embedder=StubEmbedder(dim=32)) as session:
        from mindpalace.index import db

        assert db.read_meta(session.conn) == ("stub-32", 32)
        assert session.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE doc_id = 'n_stale'"
        ).fetchone()[0] == 0


def test_an_external_config_edit_is_reloaded(tmp_path):
    with open_session(tmp_path) as session:
        original = session.config.thresholds.abstain_cosine_floor
        session.paths.mindpalace_md.write_text(
            session.paths.mindpalace_md.read_text().replace(
                "abstain_cosine_floor: 0.35", "abstain_cosine_floor: 0.6"
            )
        )
        assert session.heal() is True
        assert session.config.thresholds.abstain_cosine_floor == 0.6
        assert original == 0.35


def test_operation_commits_on_success(tmp_path):
    with open_session(tmp_path) as session:
        with session.operation({"tool": "test"}):
            pass
        assert session.oplog.pending() == []


def test_operation_leaves_a_pending_record_on_failure(tmp_path):
    with open_session(tmp_path) as session:
        with pytest.raises(RuntimeError):
            with session.operation({"tool": "test"}):
                raise RuntimeError("boom")
        assert len(session.oplog.pending()) == 1


def test_heal_clears_pending_operations(tmp_path):
    with open_session(tmp_path) as session:
        with pytest.raises(RuntimeError):
            with session.operation({"tool": "test"}):
                raise RuntimeError("boom")

    with open_session(tmp_path) as session:
        assert session.oplog.pending() == []


def test_heal_reports_whether_it_did_work(tmp_path):
    with open_session(tmp_path) as session:
        assert session.heal() is False  # nothing pending, no drift
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.session'`

- [ ] **Step 3: Implement the session**

`mindpalace/session.py`:

```python
"""Process-level vault session: locking, crash healing, operation framing."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from mindpalace.config import Config, load_config, open_vault
from mindpalace.embed import Embedder, get_embedder
from mindpalace.index import db
from mindpalace.index.sync import has_drift, sync
from mindpalace.oplog import DecisionLog, OpLog
from mindpalace.vault.paths import VaultPaths
from mindpalace.vault.store import VaultStore


class VaultLockedError(RuntimeError):
    """Raised when another server already holds this vault."""


class Session:
    def __init__(
        self, root: Path, *, init: bool = False, embedder: Embedder | None = None
    ) -> None:
        self._root = Path(root)
        self._init = init
        self._explicit_embedder = embedder
        self._lock_fd: int | None = None
        self.paths: VaultPaths
        self.config: Config
        self.store: VaultStore
        self.embedder: Embedder
        self.opened = False

    # ---- lifecycle ----------------------------------------------------

    def open(self) -> "Session":
        self.paths, self.config = open_vault(self._root, init=self._init)
        self._acquire_lock()
        self.store = VaultStore(self.paths)
        self.conn = db.connect(self.paths.graph_db)
        db.create_schema(self.conn)
        self.oplog = OpLog(self.paths.op_log)
        self.decisions = DecisionLog(self.paths.decisions_log)
        self.embedder = self._explicit_embedder or get_embedder(self.config.embedder)
        self.opened = True
        self._verify_cache_model()
        self.heal()
        return self

    def close(self) -> None:
        if self.opened:
            self.conn.close()
            self._release_lock()
            self.opened = False

    def __enter__(self) -> "Session":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _acquire_lock(self) -> None:
        """Exclusive advisory lock held for the session's lifetime.

        An `exists()` check followed by a write would be check-then-write: two
        servers starting together could both see no lock and both proceed.
        `flock` is atomic, and the kernel releases it if we die, which also makes
        a stale lock from a crashed process self-healing.
        """
        lock = self.paths.lock
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = os.read(self._lock_fd, 32).decode(errors="replace").strip()
            os.close(self._lock_fd)
            self._lock_fd = None
            raise VaultLockedError(
                f"vault {self.paths.root} is already open (pid {holder or 'unknown'})"
            ) from exc
        os.ftruncate(self._lock_fd, 0)
        os.write(self._lock_fd, str(os.getpid()).encode())
        os.fsync(self._lock_fd)

    def _release_lock(self) -> None:
        if getattr(self, "_lock_fd", None) is None:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None
        self.paths.lock.unlink(missing_ok=True)

    def _verify_cache_model(self) -> None:
        """Rebuild when the embedder changed (spec §5).

        Vectors are filtered by `model_id`, so a swapped model does not error —
        it returns nothing, which reads as an empty vault. Detect and rebuild.
        """
        if db.read_meta(self.conn) == (self.embedder.model_id, self.embedder.dim):
            return
        self.resync()

    # ---- operations ---------------------------------------------------

    @contextmanager
    def operation(self, intent: dict):
        op_id = self.oplog.begin(intent)
        yield op_id
        self.oplog.commit(op_id)

    def heal(self) -> bool:
        """Reconcile after a crash or an external edit. Idempotent."""
        pending = self.oplog.pending()
        drifted = has_drift(self.conn, self.store)
        if not pending and not drifted:
            return False
        if drifted:
            # MINDPALACE.md is in the drift set, and its edge_types drive how the
            # fold groups aggregates — so reload before re-deriving anything.
            self.config = load_config(self.paths.mindpalace_md)
        self.resync()
        for record in pending:
            self.oplog.commit(record["op"])
        return True

    def resync(self) -> None:
        sync(self.conn, self.store, self.config, self.embedder, self.statuses())

    def statuses(self) -> dict[str, str]:
        return self.decisions.status_map()
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_session.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Write the failing write-path tools test**

`tests/test_tools_write.py`:

```python
import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import ToolError, save_capture, write_note


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def test_save_capture_writes_a_file_and_returns_its_id(session):
    result = save_capture(session, "The plateau is about data exhaustion.")
    assert result["id"].startswith("c_")
    assert (session.paths.root / result["path"]).exists()


def test_save_capture_payload_carries_the_vocabularies(session):
    result = save_capture(session, "A thought.")
    assert result["entity_types"] == session.config.entity_types
    assert result["edge_vocabulary"]["contradicts"] == "symmetric"
    assert result["edge_vocabulary"]["supports"] == "directed"


def test_save_capture_payload_carries_the_extraction_instruction(session):
    result = save_capture(session, "A thought.")
    assert "Proposing nothing is a valid outcome" in result["next"]


def test_save_capture_finds_nearest_existing_material(session):
    save_capture(session, "sourdough starter hydration ratios")
    write_note(
        session,
        derived_from=save_capture(session, "the plateau is data exhaustion")["id"],
        content="Scaling limits come from data supply, not architecture.",
    )
    result = save_capture(session, "data exhaustion and scaling limits")
    assert any("scaling" in hit["snippet"].lower() for hit in result["nearest"])


def test_two_captures_in_the_same_minute_both_survive(session):
    first = save_capture(session, "first thought")
    second = save_capture(session, "second thought")
    assert first["path"] != second["path"]
    assert (session.paths.root / first["path"]).exists()
    assert (session.paths.root / second["path"]).exists()


def test_write_note_assigns_assertion_ids_and_leaves_them_proposed(session):
    capture = save_capture(session, "The plateau is data exhaustion.")
    result = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply is the binding constraint.",
        entities=[{"name": "scaling-laws", "type": "concept", "description": "…"}],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["id"].startswith("x_")
    assert assertion["status"] == "proposed"


def test_write_note_rejects_an_unknown_edge_type(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="invented"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {
                    "source": "a",
                    "target": "b",
                    "type": "invented",
                    "strength": 5,
                    "description": "x",
                }
            ],
        )


def test_write_note_rejects_an_unknown_entity_type(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="teapot"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            entities=[{"name": "a", "type": "teapot", "description": "x"}],
        )


def test_write_note_rejects_a_missing_capture(session):
    with pytest.raises(ToolError, match="c_nope"):
        write_note(session, derived_from="c_nope", content="Body.")


def test_write_note_creates_entity_pages(session):
    capture = save_capture(session, "A thought.")
    write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        entities=[{"name": "Scaling Laws", "type": "concept", "description": "…"}],
    )
    assert session.store.read_entity_page("scaling-laws") is not None


def test_save_capture_rejects_empty_text(session):
    with pytest.raises(ToolError, match="empty"):
        save_capture(session, "   ")


def test_write_note_rejects_empty_content(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="content is empty"):
        write_note(session, derived_from=capture["id"], content="  ")


def test_write_note_rejects_a_name_that_normalises_to_nothing(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="empty slug"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            entities=[{"name": "!!!", "type": "concept", "description": "x"}],
        )


def test_write_note_rejects_out_of_range_strength(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="1-10"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {
                    "source": "a",
                    "target": "b",
                    "type": "contradicts",
                    "strength": 99,
                    "description": "x",
                }
            ],
        )


def test_write_note_reports_a_missing_field_as_a_tool_error(session):
    """A KeyError traceback tells the assistant nothing it can act on."""
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="missing 'description'"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {"source": "a", "target": "b", "type": "contradicts"}
            ],
        )
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_tools_write.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.tools'`

- [ ] **Step 7: Implement the write-path tools**

`mindpalace/tools.py`:

```python
"""Tool implementations as plain functions over a Session.

Keeping these free of MCP types makes them directly testable; server.py does
nothing but wire them to the protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mindpalace.ids import new_id, slugify
from mindpalace.models import (
    Capture,
    ClaimAssertion,
    EntityInstance,
    Note,
    RelationshipAssertion,
)
from mindpalace.rebuild import rebuild
from mindpalace.retrieve import local_search
from mindpalace.session import Session

DEFAULT_EXTRACTION_NEXT = (
    "Read any nearest notes you need, then call write_note with this capture id. "
    "Propose a relationship only where you can give a specific rationale citing "
    "both endpoints. Proposing nothing is a valid outcome."
)


class ToolError(ValueError):
    """Raised for invalid tool input. Surfaced to the assistant verbatim."""


STRENGTH_RANGE = range(1, 11)


def _require(condition: bool, message: str) -> None:
    """Every invalid input must reach the assistant as a ToolError it can act on.

    Without this, a missing dict key surfaces as a KeyError traceback and the
    assistant has no idea what to fix.
    """
    if not condition:
        raise ToolError(message)


def _require_slug(raw: str, field: str) -> str:
    """Guard against names that normalise away entirely, e.g. "!!!" -> ""."""
    slug = slugify(raw)
    _require(bool(slug), f"{field} {raw!r} normalises to an empty slug")
    return slug


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def _relative(session: Session, path) -> str:
    return str(path.relative_to(session.paths.root))


def _edge_vocabulary(session: Session) -> dict[str, str]:
    return {
        name: ("directed" if spec.directed else "symmetric")
        for name, spec in session.config.edge_types.items()
    }


def _known_entities(session: Session, limit: int = 40) -> list[dict]:
    rows = session.conn.execute(
        "SELECT slug, type, rank FROM entities ORDER BY rank DESC, slug LIMIT ?",
        (limit,),
    ).fetchall()
    entities = []
    for row in rows:
        page = session.store.read_entity_page(row["slug"])
        entities.append(
            {
                "name": row["slug"],
                "type": row["type"],
                "rank": row["rank"],
                "aliases": (page.user.get("aliases", []) if page else []),
            }
        )
    return entities


def _previously_dismissed(session: Session) -> list[dict]:
    reasons = session.decisions.dismissal_reasons()
    if not reasons:
        return []
    placeholders = ",".join("?" for _ in reasons)
    rows = session.conn.execute(
        f"SELECT id, source, type, target FROM assertions WHERE id IN ({placeholders})",
        tuple(reasons),
    ).fetchall()
    return [
        {
            "pair": f"{row['source']}|{row['type']}|{row['target']}",
            "reason": reasons[row["id"]],
        }
        for row in rows
    ]


def save_capture(
    session: Session, text: str, why: str | None = None, source: str = "manual"
) -> dict:
    if not text.strip():
        raise ToolError("capture text is empty")

    when = _now()
    capture_id = new_id("c_")
    capture = Capture(
        id=capture_id, created=_timestamp(when), source=source, why=why, text=text
    )

    with session.operation({"tool": "save_capture", "capture": capture_id}):
        path = session.store.write_capture(capture, when)
        session.resync()

    nearest = local_search(session.conn, session.embedder, text, session.config, k=5)

    return {
        "id": capture_id,
        "path": _relative(session, path),
        "nearest": nearest["hits"],
        "known_entities": _known_entities(session),
        "previously_dismissed": _previously_dismissed(session),
        "entity_types": session.config.entity_types,
        "edge_vocabulary": _edge_vocabulary(session),
        "next": session.config.templates.get(
            "extraction_next", DEFAULT_EXTRACTION_NEXT
        ),
    }


def write_note(
    session: Session,
    derived_from: str,
    content: str,
    entities: list[dict] | tuple = (),
    relationship_assertions: list[dict] | tuple = (),
    claim_assertions: list[dict] | tuple = (),
) -> dict:
    _require(bool(content.strip()), "note content is empty")

    known_captures = {capture.id for capture in session.store.iter_captures()}
    _require(derived_from in known_captures, f"no such capture: {derived_from}")

    for entity in entities:
        for field in ("name", "type", "description"):
            _require(field in entity, f"entity entry is missing {field!r}: {entity}")
        _require(
            entity["type"] in session.config.entity_types,
            f"unknown entity type {entity['type']!r}; "
            f"choose from {session.config.entity_types}",
        )
        _require_slug(entity["name"], "entity name")

    for assertion in relationship_assertions:
        for field in ("source", "target", "type", "description"):
            _require(
                field in assertion, f"relationship is missing {field!r}: {assertion}"
            )
        _require(
            assertion["type"] in session.config.edge_types,
            f"unknown edge type {assertion['type']!r}; "
            f"choose from {sorted(session.config.edge_types)}",
        )
        strength = assertion.get("strength", 5)
        _require(
            isinstance(strength, int) and strength in STRENGTH_RANGE,
            f"strength must be an integer 1-10, got {strength!r}",
        )
        _require_slug(assertion["source"], "relationship source")
        _require_slug(assertion["target"], "relationship target")

    for claim in claim_assertions:
        for field in ("subject", "text"):
            _require(field in claim, f"claim is missing {field!r}: {claim}")
        _require(bool(claim["text"].strip()), "claim text is empty")
        _require_slug(claim["subject"], "claim subject")

    note_id = new_id("n_")
    built_relationships = tuple(
        RelationshipAssertion(
            id=new_id("x_"),
            source=slugify(a["source"]),
            target=slugify(a["target"]),
            type=a["type"],
            strength=int(a.get("strength", 5)),
            description=a["description"],
        )
        for a in relationship_assertions
    )
    built_claims = tuple(
        ClaimAssertion(id=new_id("k_"), subject=slugify(c["subject"]), text=c["text"])
        for c in claim_assertions
    )
    note = Note(
        id=note_id,
        derived_from=derived_from,
        created=_timestamp(_now()),
        author="llm",
        body=content,
        entities=tuple(
            EntityInstance(
                name=slugify(e["name"]), type=e["type"], description=e["description"]
            )
            for e in entities
        ),
        relationship_assertions=built_relationships,
        claim_assertions=built_claims,
    )

    slug = slugify(content.splitlines()[0] if content.strip() else note_id)
    with session.operation({"tool": "write_note", "note": note_id}):
        path = session.store.write_note(note, slug or note_id)
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )

    return {
        "id": note_id,
        "path": _relative(session, path),
        "entities": [e.name for e in note.entities],
        "relationship_assertions": [
            {
                "id": a.id,
                "pair": f"{a.source}|{a.type}|{a.target}",
                "status": "proposed",
            }
            for a in built_relationships
        ],
        "claim_assertions": [
            {"id": c.id, "subject": c.subject, "status": "proposed"}
            for c in built_claims
        ],
        "next": "Nothing further is required. Proposals wait in review_queue.",
    }
```

- [ ] **Step 8: Run it to verify it passes**

Run: `uv run pytest tests/test_tools_write.py -v`
Expected: PASS (15 tests)

This task must end green like every other. The `previously_dismissed` behaviour
needs `resolve_assertion`, which does not exist yet, so its test lives in Task 16
rather than sitting red here.

- [ ] **Step 9: Commit**

```bash
git add mindpalace/session.py mindpalace/tools.py tests/test_session.py tests/test_tools_write.py
git commit -m "feat: vault session with locking and crash healing, plus write-path tools"
```

---

## Task 16: Read, lifecycle, prose, and clustering tools

**Files:**
- Modify: `mindpalace/models.py` (make `Note.derived_from` optional)
- Modify: `mindpalace/tools.py` (append the remaining eleven tools)
- Test: `tests/test_tools_read.py`, `tests/test_tools_lifecycle.py`

**Interfaces:**
- Consumes: `mindpalace.cluster.{partition, match_lineages, should_cluster}`, `mindpalace.citations.unresolvable`, `mindpalace.retrieve.{local_search, global_search}`
- Produces: `tools.read(session, identifier) -> dict`, `tools.neighbors(session, identifier, depth=1, edge_types=None) -> dict`, `tools.get_entity(session, name) -> dict`, `tools.graph_stats(session) -> dict`, `tools.search_local(session, query, k=8, expand_graph=False) -> dict`, `tools.search_global(session, query) -> dict`, `tools.propose_relationship(session, source, target, type, description, strength=5) -> dict`, `tools.resolve_assertion(session, identifier, action, reason=None) -> dict`, `tools.review_queue(session, limit=20) -> dict`, `tools.cluster_tool(session, force=False) -> dict`, `tools.write_community_report(session, lineage_id, title, summary, rank, findings, cites) -> dict`, `tools.write_entity_description(session, slug, description) -> dict`, `tools.rebuild_tool(session, scope="all") -> dict`

A note may legitimately have no source capture — a synthesis, or a relationship
spotted outside extraction — so `derived_from` becomes optional here.

- [ ] **Step 1: Make `derived_from` optional**

In `mindpalace/models.py`, change the `Note` field and both serialisers:

```python
@dataclass(frozen=True)
class Note:
    id: str
    derived_from: str | None
    created: str
    author: str
    body: str
    entities: tuple[EntityInstance, ...] = ()
    relationship_assertions: tuple[RelationshipAssertion, ...] = ()
    claim_assertions: tuple[ClaimAssertion, ...] = ()
```

In `note_to_markdown`, replace the unconditional `derived_from` entry:

```python
    data: dict = {"id": note.id}
    if note.derived_from is not None:
        data["derived_from"] = note.derived_from
    data["created"] = note.created
    data["author"] = note.author
```

In `note_from_markdown`, replace `derived_from=data["derived_from"]` with:

```python
        derived_from=data.get("derived_from"),
```

Add to `tests/test_models.py`:

```python
def test_note_round_trips_without_a_source_capture():
    synthesis = Note(
        id="n_03",
        derived_from=None,
        created="2026-08-08T17:00:00Z",
        author="user",
        body="A link I spotted myself.",
    )
    assert note_from_markdown(note_to_markdown(synthesis)) == synthesis
```

- [ ] **Step 2: Run the existing suite to confirm nothing regressed**

Run: `uv run pytest tests/test_models.py tests/test_fold.py tests/test_sync.py -v`
Expected: PASS

- [ ] **Step 3: Write the failing read-tools test**

`tests/test_tools_read.py`:

```python
import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import (
    ToolError,
    get_entity,
    graph_stats,
    neighbors,
    read,
    resolve_assertion,
    save_capture,
    search_local,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def populated(session):
    capture = save_capture(session, "The plateau is about data exhaustion.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply binds scaling, not architecture.",
        entities=[
            {"name": "scaling-laws", "type": "concept", "description": "Compute vs loss."},
            {"name": "data-exhaustion", "type": "concept", "description": "Running out."},
        ],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    return session, capture, note


def test_read_dispatches_on_prefix(populated):
    session, capture, note = populated
    assert "data exhaustion" in read(session, capture["id"])["text"]
    assert "Data supply" in read(session, note["id"])["text"]
    assert read(session, "e_scaling-laws")["kind"] == "entity"


def test_read_rejects_an_unknown_id(populated):
    session, _, _ = populated
    with pytest.raises(ToolError, match="not found"):
        read(session, "n_missing")


def test_neighbors_are_empty_until_confirmation(populated):
    session, _, _ = populated
    assert neighbors(session, "e_scaling-laws")["neighbours"] == []


def test_neighbors_appear_once_confirmed(populated):
    session, _, note = populated
    resolve_assertion(session, note["relationship_assertions"][0]["id"], "confirm")
    result = neighbors(session, "e_scaling-laws")
    assert result["neighbours"][0]["slug"] == "data-exhaustion"
    assert result["neighbours"][0]["type"] == "contradicts"


def test_neighbors_filter_by_edge_type(populated):
    session, _, note = populated
    resolve_assertion(session, note["relationship_assertions"][0]["id"], "confirm")
    assert neighbors(session, "e_scaling-laws", edge_types=["supports"])["neighbours"] == []


def test_get_entity_resolves_through_aliases(populated):
    session, _, _ = populated
    page = session.store.read_entity_page("scaling-laws")
    page.user = {"aliases": ["scaling law"]}
    session.store.write_entity_page(page)
    assert get_entity(session, "Scaling Law")["slug"] == "scaling-laws"


def test_get_entity_rejects_an_unknown_name(populated):
    session, _, _ = populated
    with pytest.raises(ToolError, match="no entity"):
        get_entity(session, "nonexistent")


def test_graph_stats_reports_distance_to_the_threshold(populated):
    session, _, _ = populated
    stats = graph_stats(session)
    assert stats["clustering"]["active"] is False
    assert stats["clustering"]["threshold"] == 150
    assert stats["clustering"]["remaining"] == 150 - stats["entities"]


def test_graph_stats_names_the_active_embedder(populated):
    session, _, _ = populated
    stats = graph_stats(session)
    assert stats["embedder"]["model_id"] == "stub-64"
    assert stats["embedder"]["sends_data_off_machine"] is False


def test_graph_stats_counts_orphans(populated):
    session, _, _ = populated
    assert graph_stats(session)["orphans"] == 2  # nothing confirmed yet


def test_search_local_wraps_retrieval(populated):
    session, _, _ = populated
    result = search_local(session, "data exhaustion")
    assert result["hits"]
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/test_tools_read.py -v`
Expected: FAIL with `ImportError: cannot import name 'read'`

- [ ] **Step 5: Append the read tools to `mindpalace/tools.py`**

```python
from mindpalace.citations import extract_ids, unresolvable
from mindpalace.cluster import Community, match_lineages, partition, should_cluster
from mindpalace.ids import UnknownIdError, id_kind
from mindpalace.rebuild import (
    community_input_hash,
    entity_input_hash,
    mark_stale_reports,
)
from mindpalace.retrieve import global_search
from mindpalace.graph.fold import fold
from mindpalace.models import CommunityReport


def _tables(session: Session):
    """Fold the current source set. Several tools need it for evidence hashing."""
    notes = list(session.store.iter_notes())
    return notes, {note.id: note for note in notes}, fold(
        notes, session.statuses(), session.config
    )


def read(session: Session, identifier: str) -> dict:
    try:
        kind = id_kind(identifier)
    except UnknownIdError as exc:
        raise ToolError(str(exc)) from exc

    if kind == "capture":
        for capture in session.store.iter_captures():
            if capture.id == identifier:
                return {"id": identifier, "kind": "capture", "text": capture.text}
    elif kind == "note":
        for note in session.store.iter_notes():
            if note.id == identifier:
                return {"id": identifier, "kind": "note", "text": note.body}
    elif kind == "entity":
        page = session.store.read_entity_page(identifier.removeprefix("e_"))
        if page is not None:
            return {
                "id": identifier,
                "kind": "entity",
                "text": page.description,
                "stale": page.stale,
                "related": page.related,
                "user": page.user,
            }
    elif kind == "community":
        report = session.store.read_report(identifier)
        if report is not None:
            return {
                "id": identifier,
                "kind": "community",
                "title": report.title,
                "text": report.summary,
                "findings": report.findings,
                "stale": report.stale,
            }
    raise ToolError(f"{identifier} not found")


def neighbors(
    session: Session,
    identifier: str,
    depth: int = 1,
    edge_types: list[str] | None = None,
) -> dict:
    frontier = {identifier.removeprefix("e_")}
    seen = set(frontier)
    collected: list[dict] = []

    for _ in range(max(1, depth)):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        rows = session.conn.execute(
            f"SELECT source, target, type, weight FROM aggregates "
            f"WHERE traversable = 1 AND (source IN ({placeholders}) "
            f"OR target IN ({placeholders}))",
            (*frontier, *frontier),
        ).fetchall()

        next_frontier: set[str] = set()
        for row in rows:
            if edge_types and row["type"] not in edge_types:
                continue
            for slug in (row["source"], row["target"]):
                if slug in seen:
                    continue
                seen.add(slug)
                next_frontier.add(slug)
                collected.append(
                    {"slug": slug, "type": row["type"], "weight": row["weight"]}
                )
        frontier = next_frontier

    grouped: dict[str, list[dict]] = {}
    for neighbour in collected:
        grouped.setdefault(neighbour["type"], []).append(neighbour)

    return {"id": identifier, "neighbours": collected, "by_type": grouped}


def _resolve_slug(session: Session, name: str) -> str | None:
    slug = slugify(name)
    row = session.conn.execute(
        "SELECT slug FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    if row is not None:
        return row["slug"]
    for page in session.store.iter_entity_pages():
        aliases = {slugify(alias) for alias in page.user.get("aliases", [])}
        if slug in aliases:
            return page.slug
    return None


def get_entity(session: Session, name: str) -> dict:
    slug = _resolve_slug(session, name)
    if slug is None:
        raise ToolError(f"no entity matching {name!r}")

    page = session.store.read_entity_page(slug)
    row = session.conn.execute(
        "SELECT type, rank FROM entities WHERE slug = ?", (slug,)
    ).fetchone()
    claims = [
        dict(record)
        for record in session.conn.execute(
            "SELECT id, text, status FROM claims WHERE subject = ?", (slug,)
        )
    ]
    return {
        "slug": slug,
        "type": row["type"] if row else (page.type if page else "unknown"),
        "rank": row["rank"] if row else 0,
        "description": page.description if page else "",
        "stale": page.stale if page else True,
        "user": page.user if page else {},
        "claims": claims,
        "neighbours": neighbors(session, f"e_{slug}")["neighbours"],
    }


def search_local(
    session: Session, query: str, k: int = 8, expand_graph: bool = False
) -> dict:
    return local_search(
        session.conn, session.embedder, query, session.config, k, expand_graph
    )


def search_global(session: Session, query: str) -> dict:
    return global_search(
        session.conn, session.store, session.embedder, query, session.config
    )


def graph_stats(session: Session) -> dict:
    def count(sql: str) -> int:
        return session.conn.execute(sql).fetchone()[0]

    entities = count("SELECT COUNT(*) FROM entities")
    threshold = session.config.thresholds.cluster_activation_entities
    communities = count("SELECT COUNT(*) FROM communities")
    stale_pages = sum(1 for page in session.store.iter_entity_pages() if page.stale)
    stale_reports = sum(1 for report in session.store.iter_reports() if report.stale)

    return {
        "entities": entities,
        "orphans": count("SELECT COUNT(*) FROM entities WHERE rank = 0"),
        "assertions": count("SELECT COUNT(*) FROM assertions"),
        "aggregates": count("SELECT COUNT(*) FROM aggregates"),
        "traversable_aggregates": count(
            "SELECT COUNT(*) FROM aggregates WHERE traversable = 1"
        ),
        "claims": count("SELECT COUNT(*) FROM claims"),
        "communities": communities,
        "stale_entity_pages": stale_pages,
        "stale_reports": stale_reports,
        "vault_issues": count("SELECT COUNT(*) FROM vault_issues"),
        "clustering": {
            "active": communities > 0,
            "eligible": should_cluster(entities, threshold),
            "entity_count": entities,
            "threshold": threshold,
            "remaining": max(0, threshold - entities),
        },
        "embedder": {
            "model_id": session.embedder.model_id,
            "kind": session.config.embedder.get("kind"),
            "sends_data_off_machine": session.config.embedder.get("kind") == "cloud",
        },
    }
```

- [ ] **Step 6: Run it to verify it passes**

Run: `uv run pytest tests/test_tools_read.py -v`
Expected: PASS (11 tests)

- [ ] **Step 7: Write the failing lifecycle test**

`tests/test_tools_lifecycle.py`:

```python
import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import (
    ToolError,
    cluster_tool,
    propose_relationship,
    resolve_assertion,
    review_queue,
    save_capture,
    write_community_report,
    write_entity_description,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def with_assertion(session):
    capture = save_capture(session, "The plateau is about data exhaustion.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply binds scaling.",
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    return session, note["relationship_assertions"][0]["id"]


def test_propose_relationship_creates_a_proposed_assertion(session):
    result = propose_relationship(
        session, "a", "b", "relates-to", "spotted later", strength=6
    )
    assert result["status"] == "proposed"
    assert result["id"].startswith("x_")


def test_propose_relationship_rejects_an_unknown_type(session):
    with pytest.raises(ToolError, match="invented"):
        propose_relationship(session, "a", "b", "invented", "x")


def test_resolve_assertion_confirms(with_assertion):
    session, assertion_id = with_assertion
    assert resolve_assertion(session, assertion_id, "confirm")["status"] == "confirmed"


def test_resolve_assertion_dismisses(with_assertion):
    session, assertion_id = with_assertion
    result = resolve_assertion(session, assertion_id, "dismiss", reason="wrong sense")
    assert result["status"] == "dismissed"


def test_a_dismissal_is_reversible_by_explicit_human_action(with_assertion):
    """Spec §7.3: dismissal is terminal for the proposal loop, not for the human.
    Forcing a whole new assertion to undo a mis-click buys no safety, and every
    flip is recorded in the decision log anyway."""
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "dismiss", reason="wrong sense")
    assert resolve_assertion(session, assertion_id, "confirm")["status"] == "confirmed"
    actions = [entry.action for entry in session.decisions.entries()]
    assert actions == ["dismiss", "confirm"]


def test_previously_dismissed_pairs_are_surfaced_not_suppressed(session):
    """Moved here from Task 15: it needs resolve_assertion to exist."""
    capture = save_capture(session, "A thought.")
    result = write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "chinchilla",
                "type": "contradicts",
                "strength": 5,
                "description": "x",
            }
        ],
    )
    resolve_assertion(
        session,
        result["relationship_assertions"][0]["id"],
        "dismiss",
        reason="different sense",
    )

    payload = save_capture(session, "Another thought about scaling.")
    dismissed = payload["previously_dismissed"]
    assert dismissed[0]["pair"] == "scaling-laws|contradicts|chinchilla"
    assert dismissed[0]["reason"] == "different sense"


def test_resolve_assertion_rejects_an_unknown_action(with_assertion):
    session, assertion_id = with_assertion
    with pytest.raises(ToolError, match="action"):
        resolve_assertion(session, assertion_id, "maybe")


def test_resolve_assertion_rejects_an_unknown_id(session):
    with pytest.raises(ToolError, match="x_ghost"):
        resolve_assertion(session, "x_ghost", "confirm")


def test_review_queue_separates_proposals_from_issues(with_assertion):
    session, _ = with_assertion
    (session.paths.notes / "broken.md").write_text("---\nnot: [closed\n")
    session.resync()

    queue = review_queue(session)
    assert len(queue["proposals"]) == 1
    assert queue["proposals"][0]["pair"].startswith("scaling-laws|contradicts")
    assert any("broken.md" in issue["path"] for issue in queue["vault_issues"])


def test_review_queue_inlines_both_endpoint_snippets(with_assertion):
    """Judging a proposal must not require extra `read` calls."""
    session, _ = with_assertion
    write_entity_description(session, "scaling-laws", "Compute, data, and loss.")
    write_entity_description(session, "data-exhaustion", "Running out of tokens.")

    [proposal] = review_queue(session)["proposals"]
    assert proposal["source_snippet"] == "Compute, data, and loss."
    assert proposal["target_snippet"] == "Running out of tokens."


def test_review_queue_ranks_by_asserted_strength(session):
    capture = save_capture(session, "A thought.")
    write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "a",
                "target": "b",
                "type": "relates-to",
                "strength": 2,
                "description": "weak",
            },
            {
                "source": "c",
                "target": "d",
                "type": "relates-to",
                "strength": 9,
                "description": "strong",
            },
        ],
    )
    strengths = [p["strength"] for p in review_queue(session)["proposals"]]
    assert strengths == [9, 2]


def test_review_queue_drops_resolved_proposals(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    assert review_queue(session)["proposals"] == []


def test_cluster_refuses_below_the_threshold(with_assertion):
    session, _ = with_assertion
    result = cluster_tool(session)
    assert result["clustered"] is False
    assert "threshold" in result["note"]


def test_cluster_with_force_partitions_a_small_graph(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    result = cluster_tool(session, force=True)
    assert result["clustered"] is True
    assert result["communities"]
    assert result["communities"][0]["members"]


def test_write_entity_description_clears_staleness(with_assertion):
    session, _ = with_assertion
    write_entity_description(session, "scaling-laws", "Compute, data, and loss.")
    page = session.store.read_entity_page("scaling-laws")
    assert page.description == "Compute, data, and loss."
    assert page.stale is False


def test_write_community_report_rejects_unresolvable_citations(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="e_ghost"):
        write_community_report(
            session,
            lineage_id,
            title="Bogus",
            summary="Cites something that does not exist.",
            rank=5.0,
            findings=[],
            cites=["e_ghost"],
        )


def test_write_community_report_stores_a_valid_report(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    write_community_report(
        session,
        lineage_id,
        title="Scaling Debate",
        summary="A cluster about scaling limits.",
        rank=6.0,
        findings=[
            {
                "summary": "Data supply dominates",
                "explanation": "Both endpoints agree [Data: Entities (e_scaling-laws)].",
            }
        ],
        cites=["e_scaling-laws"],
    )
    stored = session.store.read_report(lineage_id)
    assert stored.title == "Scaling Debate"
    assert stored.stale is False


def test_write_community_report_requires_every_finding_to_be_grounded(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="no \\[Data"):
        write_community_report(
            session,
            lineage_id,
            title="Ungrounded",
            summary="Reads as evidence but cites nothing.",
            rank=5.0,
            findings=[{"summary": "A confident claim", "explanation": "Trust me."}],
            cites=["e_scaling-laws"],
        )


def test_write_community_report_rejects_a_citation_to_a_missing_note(with_assertion):
    """Note and capture ids used to be waved through without checking."""
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    lineage_id = session.conn.execute(
        "SELECT lineage_id FROM communities LIMIT 1"
    ).fetchone()[0]

    with pytest.raises(ToolError, match="n_nonexistent"):
        write_community_report(
            session,
            lineage_id,
            title="Bogus",
            summary="Cites a note that was never written.",
            rank=5.0,
            findings=[],
            cites=["n_nonexistent"],
        )


def test_reclustering_preserves_report_lineage(with_assertion):
    session, assertion_id = with_assertion
    resolve_assertion(session, assertion_id, "confirm")
    cluster_tool(session, force=True)
    before = session.conn.execute("SELECT lineage_id FROM communities").fetchone()[0]
    cluster_tool(session, force=True)
    after = session.conn.execute("SELECT lineage_id FROM communities").fetchone()[0]
    assert before == after


def test_reclustering_marks_an_affected_report_stale_immediately(session):
    """Otherwise global_search serves an obsolete report as `present` until some
    unrelated rebuild happens to run."""
    capture = save_capture(session, "A thought.")
    note = write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        relationship_assertions=[
            {
                "source": "a",
                "target": "b",
                "type": "contradicts",
                "strength": 5,
                "description": "x",
            }
        ],
    )
    resolve_assertion(session, note["relationship_assertions"][0]["id"], "confirm")
    clustered = cluster_tool(session, force=True)
    lineage_id = clustered["communities"][0]["lineage_id"]
    write_community_report(
        session,
        lineage_id,
        title="Pair",
        summary="Written from the original evidence.",
        rank=5.0,
        findings=[],
        cites=["e_a"],
    )
    assert session.store.read_report(lineage_id).stale is False

    second = write_note(
        session,
        derived_from=save_capture(session, "More.")["id"],
        content="Body two.",
        relationship_assertions=[
            {
                "source": "a",
                "target": "b",
                "type": "contradicts",
                "strength": 9,
                "description": "reinforced",
            }
        ],
    )
    resolve_assertion(session, second["relationship_assertions"][0]["id"], "confirm")

    result = cluster_tool(session, force=True)
    assert result["reports_marked_stale"] == 1
    assert session.store.read_report(lineage_id).stale is True
```

- [ ] **Step 8: Run it to verify it fails**

Run: `uv run pytest tests/test_tools_lifecycle.py -v`
Expected: FAIL with `ImportError: cannot import name 'cluster_tool'`

- [ ] **Step 9: Append the lifecycle tools to `mindpalace/tools.py`**

```python
VALID_ACTIONS = {"confirm": "confirmed", "dismiss": "dismissed"}


def propose_relationship(
    session: Session,
    source: str,
    target: str,
    type: str,
    description: str,
    strength: int = 5,
) -> dict:
    _require(
        type in session.config.edge_types,
        f"unknown edge type {type!r}; choose from {sorted(session.config.edge_types)}",
    )
    _require(
        isinstance(strength, int) and strength in STRENGTH_RANGE,
        f"strength must be an integer 1-10, got {strength!r}",
    )
    _require(bool(description.strip()), "a proposed relationship needs a rationale")

    assertion = RelationshipAssertion(
        id=new_id("x_"),
        source=_require_slug(source, "relationship source"),
        target=_require_slug(target, "relationship target"),
        type=type,
        strength=int(strength),
        description=description,
    )
    note = Note(
        id=new_id("n_"),
        derived_from=None,
        created=_timestamp(_now()),
        author="user",
        body=f"Relationship proposed outside extraction: {description}",
        relationship_assertions=(assertion,),
    )
    with session.operation({"tool": "propose_relationship", "note": note.id}):
        session.store.write_note(note, f"link-{assertion.source}-{assertion.target}")
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )
    return {
        "id": assertion.id,
        "note": note.id,
        "pair": f"{assertion.source}|{assertion.type}|{assertion.target}",
        "status": "proposed",
    }


def resolve_assertion(
    session: Session, identifier: str, action: str, reason: str | None = None
) -> dict:
    if action not in VALID_ACTIONS:
        raise ToolError(f"action must be 'confirm' or 'dismiss', got {action!r}")

    exists = session.conn.execute(
        "SELECT 1 FROM assertions WHERE id = ? UNION SELECT 1 FROM claims WHERE id = ?",
        (identifier, identifier),
    ).fetchone()
    if exists is None:
        raise ToolError(f"no such assertion: {identifier}")

    with session.operation(
        {"tool": "resolve_assertion", "assertion": identifier, "action": action}
    ) as op_id:
        session.decisions.append(identifier, action, "resolve_assertion", op_id, reason)
        rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
        )

    return {"id": identifier, "status": VALID_ACTIONS[action], "reason": reason}


def _endpoint_snippet(session: Session, slug: str) -> str:
    """One line describing an entity, so a reviewer needs no extra `read` call."""
    page = session.store.read_entity_page(slug)
    if page is not None and page.description.strip():
        return page.description.strip().splitlines()[0][:200]
    row = session.conn.execute(
        "SELECT substr(text, 1, 200) AS snippet FROM docs WHERE doc_id = ?",
        (f"e_{slug}",),
    ).fetchone()
    if row is not None and row["snippet"]:
        return row["snippet"]
    return "(no description written yet)"


def review_queue(session: Session, limit: int = 20) -> dict:
    """Proposals are ranked by asserted strength.

    Not by "retrieval score": there is no query here to score anything against,
    so such a number would have nothing behind it.
    """
    proposals = [
        {
            "id": row["id"],
            "kind": "relationship",
            "pair": f"{row['source']}|{row['type']}|{row['target']}",
            "strength": row["strength"],
            "why": row["description"],
            "note": row["note_id"],
            "source_snippet": _endpoint_snippet(session, row["source"]),
            "target_snippet": _endpoint_snippet(session, row["target"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, source, target, type, strength, description "
            "FROM assertions WHERE status = 'proposed' "
            "ORDER BY strength DESC, id LIMIT ?",
            (limit,),
        )
    ]
    proposals += [
        {
            "id": row["id"],
            "kind": "claim",
            "subject": row["subject"],
            "text": row["text"],
            "note": row["note_id"],
            "subject_snippet": _endpoint_snippet(session, row["subject"]),
        }
        for row in session.conn.execute(
            "SELECT id, note_id, subject, text FROM claims "
            "WHERE status = 'proposed' ORDER BY id LIMIT ?",
            (limit,),
        )
    ]

    issues = [
        {"path": row["path"], "kind": row["kind"], "detail": row["detail"]}
        for row in session.conn.execute(
            "SELECT path, kind, detail FROM vault_issues LIMIT ?", (limit,)
        )
    ]
    issues += [
        {"path": f"entities/{page.slug}.md", "kind": "stale_prose", "detail": "needs rewrite"}
        for page in session.store.iter_entity_pages()
        if page.stale
    ]
    issues += [
        {
            "path": f"communities/{report.lineage_id}",
            "kind": "stale_report",
            "detail": "membership changed since this was written",
        }
        for report in session.store.iter_reports()
        if report.stale
    ]

    return {"proposals": proposals, "vault_issues": issues}


def _stored_communities(session: Session) -> list[Community]:
    return [
        Community(
            lineage_id=row["lineage_id"],
            level=row["level"],
            members=frozenset(row["members"].split(",")) if row["members"] else frozenset(),
            parent=row["parent"],
        )
        for row in session.conn.execute(
            "SELECT lineage_id, level, parent, members FROM communities"
        )
    ]


def cluster_tool(session: Session, force: bool = False) -> dict:
    stats = graph_stats(session)
    threshold = session.config.thresholds.cluster_activation_entities
    if not force and not should_cluster(stats["entities"], threshold):
        return {
            "clustered": False,
            "communities": [],
            "note": (
                f"{stats['entities']} entities; clustering activates at the "
                f"threshold of {threshold}. Pass force=true to run anyway."
            ),
        }

    _, _, tables = _tables(session)
    fresh = partition(tables.aggregates, session.config)
    matched = match_lineages(
        fresh,
        _stored_communities(session),
        session.config.thresholds.community_lineage_jaccard,
    )

    with session.operation({"tool": "cluster"}):
        with session.conn:
            session.conn.execute("DELETE FROM communities")
            session.conn.executemany(
                "INSERT INTO communities (lineage_id, level, parent, members) "
                "VALUES (?, ?, ?, ?)",
                [
                    (c.lineage_id, c.level, c.parent, ",".join(sorted(c.members)))
                    for c in matched
                ],
            )
        # Reclustering must flip affected reports to stale immediately. Deferring
        # it to the next rebuild would let global_search serve an obsolete report
        # as `present` in the meantime.
        stale_reports = mark_stale_reports(session.conn, session.store, tables)

    payload = []
    for community in matched:
        report = session.store.read_report(community.lineage_id)
        expected = community_input_hash(sorted(community.members), tables)
        payload.append(
            {
                "lineage_id": community.lineage_id,
                "level": community.level,
                "members": sorted(community.members),
                "needs_report": report is None or report.input_hash != expected,
                "aggregates": [
                    {
                        "pair": f"{a.source}|{a.type}|{a.target}",
                        "weight": a.weight,
                    }
                    for a in tables.aggregates.values()
                    if a.traversable
                    and a.source in community.members
                    and a.target in community.members
                ],
            }
        )

    return {
        "clustered": True,
        "communities": payload,
        "reports_marked_stale": stale_reports,
        "instructions": session.config.templates.get("report_next", ""),
    }


def write_community_report(
    session: Session,
    lineage_id: str,
    title: str,
    summary: str,
    rank: float,
    findings: list[dict],
    cites: list[str],
) -> dict:
    row = session.conn.execute(
        "SELECT members FROM communities WHERE lineage_id = ?", (lineage_id,)
    ).fetchone()
    if row is None:
        raise ToolError(f"no such community: {lineage_id}")

    def resolves(identifier: str) -> bool:
        """Every cited id must point at something that exists — including notes
        and captures, which the first version waved through unchecked."""
        try:
            kind = id_kind(identifier)
        except UnknownIdError:
            return False
        if kind == "entity":
            return (
                session.conn.execute(
                    "SELECT 1 FROM entities WHERE slug = ?",
                    (identifier.removeprefix("e_"),),
                ).fetchone()
                is not None
            )
        if kind == "relationship_assertion":
            table = "assertions"
        elif kind == "claim_assertion":
            table = "claims"
        else:  # note, capture, community — all indexed as documents
            return (
                session.conn.execute(
                    "SELECT 1 FROM docs WHERE doc_id = ?", (identifier,)
                ).fetchone()
                is not None
            )
        return (
            session.conn.execute(
                f"SELECT 1 FROM {table} WHERE id = ?", (identifier,)
            ).fetchone()
            is not None
        )

    cited = list(cites) + extract_ids(summary)
    for index, finding in enumerate(findings):
        grounded = extract_ids(finding.get("summary", "")) + extract_ids(
            finding.get("explanation", "")
        )
        _require(
            bool(grounded),
            f"finding {index} carries no [Data: …] citation; every finding must "
            f"name the evidence it rests on",
        )
        cited.extend(grounded)

    _require(bool(cited), "a report must cite the evidence it was written from")
    missing = unresolvable(list(dict.fromkeys(cited)), resolves)
    _require(not missing, f"report cites unresolvable ids: {missing}")

    members = row["members"].split(",") if row["members"] else []
    _, _, tables = _tables(session)
    report = CommunityReport(
        lineage_id=lineage_id,
        level=0,
        title=title,
        summary=summary,
        rank=float(rank),
        findings=findings,
        cites=list(dict.fromkeys(cited)),
        generated_from=members,
        input_hash=community_input_hash(members, tables),
        stale=False,
    )
    with session.operation({"tool": "write_community_report", "community": lineage_id}):
        session.store.write_report(report)
        session.resync()
    return {"lineage_id": lineage_id, "stale": False}


def write_entity_description(session: Session, slug: str, description: str) -> dict:
    slug = slugify(slug)
    page = session.store.read_entity_page(slug)
    if page is None:
        raise ToolError(f"no entity page for {slug!r}; run rebuild first")

    _require(bool(description.strip()), "an entity description cannot be empty")

    _, notes_by_id, tables = _tables(session)
    entity = tables.entities.get(slug)
    page.description = description
    page.stale = False
    if entity is not None:
        page.input_hash = entity_input_hash(entity, tables, notes_by_id)

    with session.operation({"tool": "write_entity_description", "entity": slug}):
        session.store.write_entity_page(page)
        session.resync()
    return {"slug": slug, "stale": False}


def rebuild_tool(session: Session, scope: str = "all") -> dict:
    if scope not in {"cache", "related_blocks", "all"}:
        raise ToolError(f"scope must be cache, related_blocks, or all; got {scope!r}")
    with session.operation({"tool": "rebuild", "scope": scope}):
        report = rebuild(
            session.conn,
            session.store,
            session.config,
            session.embedder,
            session.statuses(),
            scope=scope,
        )
    return {
        "scope": scope,
        "notes_synced": report.synced,
        "pages_written": report.pages_written,
        "pages_marked_stale": report.pages_marked_stale,
        "reports_marked_stale": report.reports_marked_stale,
    }
```

Add `extract_ids` to the `mindpalace.citations` import at the top of the file:

```python
from mindpalace.citations import extract_ids, unresolvable
```

- [ ] **Step 10: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS — including the previously-failing
`test_previously_dismissed_pairs_are_surfaced_not_suppressed` from Task 15

- [ ] **Step 11: Commit**

```bash
git add mindpalace/models.py mindpalace/tools.py tests/
git commit -m "feat: read, lifecycle, prose, and clustering tools"
```

---

## Task 17: MCP wiring, CLI, and the end-to-end invariants

**Files:**
- Create: `mindpalace/server.py`
- Test: `tests/test_server.py`, `tests/test_integration.py`

**Interfaces:**
- Consumes: `mindpalace.tools` (all fifteen), `mindpalace.session.Session`
- Produces: `build_server(session) -> FastMCP`, `main(argv=None) -> int`, `TOOL_NAMES: tuple[str, ...]`

- [ ] **Step 1: Write the failing server test**

`tests/test_server.py`:

```python
import asyncio

import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.server import TOOL_NAMES, build_server, main
from mindpalace.session import Session


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def registered_names(session):
    server = build_server(session)
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_all_fifteen_tools_are_registered(session):
    assert registered_names(session) == set(TOOL_NAMES)
    assert len(TOOL_NAMES) == 15


def test_every_tool_has_a_description(session):
    server = build_server(session)
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"


def test_main_requires_a_vault_argument(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_main_reports_a_refused_vault_without_traceback(tmp_path, capsys):
    (tmp_path / "unrelated.txt").write_text("hello")
    assert main(["--vault", str(tmp_path), "--init", "--check"]) == 2
    assert "not a Mind Palace vault" in capsys.readouterr().err


def test_main_check_succeeds_on_a_fresh_vault(tmp_path):
    assert main(["--vault", str(tmp_path), "--init", "--check"]) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindpalace.server'`

- [ ] **Step 3: Implement the server and CLI**

`mindpalace/server.py`:

```python
"""MCP wiring. All logic lives in tools.py; this module only exposes it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mindpalace import tools
from mindpalace.config import ConfigError
from mindpalace.embed import EmbedderError
from mindpalace.session import Session, VaultLockedError

TOOL_NAMES = (
    "save_capture",
    "write_note",
    "local_search",
    "global_search",
    "read",
    "neighbors",
    "get_entity",
    "graph_stats",
    "propose_relationship",
    "resolve_assertion",
    "review_queue",
    "cluster",
    "write_community_report",
    "write_entity_description",
    "rebuild",
)


def build_server(session: Session) -> FastMCP:
    server = FastMCP("mindpalace")

    @server.tool(name="save_capture")
    def _save_capture(text: str, why: str | None = None, source: str = "manual") -> dict:
        """Save a thought verbatim and return the nearest existing material,
        the known entities, and what to extract next."""
        return tools.save_capture(session, text, why, source)

    @server.tool(name="write_note")
    def _write_note(
        derived_from: str,
        content: str,
        entities: list[dict] | None = None,
        relationship_assertions: list[dict] | None = None,
        claim_assertions: list[dict] | None = None,
    ) -> dict:
        """Record your analysis of a capture plus the entities, relationships,
        and claims you extracted. All assertions enter as proposed."""
        return tools.write_note(
            session,
            derived_from,
            content,
            entities or [],
            relationship_assertions or [],
            claim_assertions or [],
        )

    @server.tool(name="local_search")
    def _local_search(query: str, k: int = 8, expand_graph: bool = False) -> dict:
        """Hybrid keyword and semantic search over captures, notes, and entities.
        Abstains when the vault holds nothing relevant."""
        return tools.search_local(session, query, k, expand_graph)

    @server.tool(name="global_search")
    def _global_search(query: str) -> dict:
        """Answer a question about themes using community reports. Explains
        itself instead of failing when clustering is not yet active."""
        return tools.search_global(session, query)

    @server.tool(name="read")
    def _read(identifier: str) -> dict:
        """Read any capture, note, entity page, or community report by id."""
        return tools.read(session, identifier)

    @server.tool(name="neighbors")
    def _neighbors(
        identifier: str, depth: int = 1, edge_types: list[str] | None = None
    ) -> dict:
        """Traverse confirmed relationships out from a node, grouped by type."""
        return tools.neighbors(session, identifier, depth, edge_types)

    @server.tool(name="get_entity")
    def _get_entity(name: str) -> dict:
        """Fetch an entity by name or alias, with its claims and neighbours."""
        return tools.get_entity(session, name)

    @server.tool(name="graph_stats")
    def _graph_stats() -> dict:
        """Counts, orphans, stale prose, clustering distance to threshold, and
        which embedder is active."""
        return tools.graph_stats(session)

    @server.tool(name="propose_relationship")
    def _propose_relationship(
        source: str, target: str, type: str, description: str, strength: int = 5
    ) -> dict:
        """Propose a link spotted outside extraction. Enters as proposed."""
        return tools.propose_relationship(
            session, source, target, type, description, strength
        )

    @server.tool(name="resolve_assertion")
    def _resolve_assertion(
        identifier: str, action: str, reason: str | None = None
    ) -> dict:
        """Confirm or dismiss a proposed relationship or claim. Call this only
        on the user's explicit instruction; the decision is logged with your
        stated reason."""
        return tools.resolve_assertion(session, identifier, action, reason)

    @server.tool(name="review_queue")
    def _review_queue(limit: int = 20) -> dict:
        """Pending proposals and vault issues, in separate sections."""
        return tools.review_queue(session, limit)

    @server.tool(name="cluster")
    def _cluster(force: bool = False) -> dict:
        """Partition the graph into communities and return those needing
        reports. Refuses below the activation threshold unless forced."""
        return tools.cluster_tool(session, force)

    @server.tool(name="write_community_report")
    def _write_community_report(
        lineage_id: str,
        title: str,
        summary: str,
        rank: float,
        findings: list[dict],
        cites: list[str],
    ) -> dict:
        """Store a community report. Rejected if any citation does not resolve."""
        return tools.write_community_report(
            session, lineage_id, title, summary, rank, findings, cites
        )

    @server.tool(name="write_entity_description")
    def _write_entity_description(slug: str, description: str) -> dict:
        """Store an entity's aggregated description and clear its stale flag."""
        return tools.write_entity_description(session, slug, description)

    @server.tool(name="rebuild")
    def _rebuild(scope: str = "all") -> dict:
        """Regenerate the cache and mark generated prose stale where its inputs
        moved. Never deletes or rewrites prose."""
        return tools.rebuild_tool(session, scope)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mindpalace")
    parser.add_argument("--vault", required=True, help="path to the vault directory")
    parser.add_argument("--init", action="store_true", help="scaffold a new vault")
    parser.add_argument(
        "--check", action="store_true", help="open and validate, then exit"
    )
    args = parser.parse_args(argv)

    try:
        session = Session(Path(args.vault), init=args.init).open()
    except (ConfigError, VaultLockedError, EmbedderError) as exc:
        # EmbedderError included so a misconfigured embedder is a legible message
        # rather than a traceback out of the MCP entry point.
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.check:
            return 0
        build_server(session).run()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the failing integration and invariant test**

`tests/test_integration.py`:

```python
import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.models import Capture
from mindpalace.session import Session
from mindpalace.tools import (
    cluster_tool,
    get_entity,
    graph_stats,
    neighbors,
    rebuild_tool,
    resolve_assertion,
    review_queue,
    save_capture,
    search_global,
    search_local,
    write_community_report,
    write_entity_description,
    write_note,
)


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def seed(session):
    """Two linked notes, one confirmed relationship."""
    first = save_capture(session, "The plateau talk is mostly about data exhaustion.")
    note = write_note(
        session,
        derived_from=first["id"],
        content="Data supply binds scaling, not architecture.",
        entities=[
            {"name": "scaling-laws", "type": "concept", "description": "Compute vs loss."},
            {"name": "data-exhaustion", "type": "concept", "description": "Running out."},
        ],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "Plateau is a supply constraint, not a ceiling.",
            }
        ],
        claim_assertions=[
            {"subject": "scaling-laws", "text": "The plateau reflects data exhaustion."}
        ],
    )
    resolve_assertion(session, note["relationship_assertions"][0]["id"], "confirm")
    return note


def snapshot(session) -> dict:
    """Everything a query could observe, for the derivability invariant."""
    def rows(sql):
        return [tuple(row) for row in session.conn.execute(sql)]

    return {
        "entities": rows("SELECT slug, type, rank FROM entities ORDER BY slug"),
        "aggregates": rows(
            "SELECT key, weight, mean_strength, traversable FROM aggregates ORDER BY key"
        ),
        "assertions": rows("SELECT id, status FROM assertions ORDER BY id"),
        "claims": rows("SELECT id, status FROM claims ORDER BY id"),
        "docs": rows("SELECT doc_id, kind FROM docs ORDER BY doc_id"),
        "search": [
            hit["id"] for hit in search_local(session, "data exhaustion")["hits"]
        ],
        "neighbours": neighbors(session, "e_scaling-laws")["neighbours"],
    }


def test_full_cycle_from_capture_to_global_answer(session):
    seed(session)

    assert review_queue(session)["proposals"][0]["kind"] == "claim"
    assert search_local(session, "data exhaustion")["hits"]
    assert neighbors(session, "e_scaling-laws")["neighbours"][0]["slug"] == "data-exhaustion"
    assert get_entity(session, "scaling-laws")["rank"] == 1

    clustered = cluster_tool(session, force=True)
    assert clustered["clustered"] is True
    lineage_id = clustered["communities"][0]["lineage_id"]

    write_community_report(
        session,
        lineage_id,
        title="Scaling and Data Supply",
        summary="A tension between architectural and data-supply accounts.",
        rank=7.0,
        findings=[
            {
                "summary": "Data supply is the binding constraint",
                "explanation": "Both notes agree [Data: Entities (e_scaling-laws)].",
            }
        ],
        cites=["e_scaling-laws", "e_data-exhaustion"],
    )

    answer = search_global(session, "what are the themes here?")
    assert answer["available"] is True
    assert answer["communities"][0]["report"] == "present"


def test_derived_cache_is_rebuildable_from_source(session):
    """The invariant: Tier 3 is a pure function of Tier 1."""
    seed(session)
    before = snapshot(session)

    session.conn.close()
    session.paths.graph_db.unlink()
    for page_path in session.paths.entities.glob("*.md"):
        text = page_path.read_text()
        head, _, _ = text.partition("<!-- mindpalace:related -->")
        page_path.write_text(head)

    from mindpalace.index import db

    session.conn = db.connect(session.paths.graph_db)
    db.create_schema(session.conn)
    rebuild_tool(session, "all")

    assert snapshot(session) == before


def test_rebuild_never_destroys_generated_prose(session):
    """Tier 2 is replaceable, not derivable — rebuild must not touch it."""
    seed(session)
    write_entity_description(session, "scaling-laws", "Prose no rebuild can recreate.")
    clustered = cluster_tool(session, force=True)
    write_community_report(
        session,
        clustered["communities"][0]["lineage_id"],
        title="Scaling and Data Supply",
        summary="Report prose that must survive.",
        rank=7.0,
        findings=[],
        cites=["e_scaling-laws"],
    )

    rebuild_tool(session, "all")

    assert (
        session.store.read_entity_page("scaling-laws").description
        == "Prose no rebuild can recreate."
    )
    reports = list(session.store.iter_reports())
    assert reports[0].summary == "Report prose that must survive."


def test_user_overrides_survive_a_full_rebuild(session):
    seed(session)
    page = session.store.read_entity_page("scaling-laws")
    page.user = {"aliases": ["scaling law"], "type": "theme"}
    session.store.write_entity_page(page)

    rebuild_tool(session, "all")

    survivor = session.store.read_entity_page("scaling-laws")
    assert survivor.user == {"aliases": ["scaling law"], "type": "theme"}
    assert get_entity(session, "Scaling Law")["slug"] == "scaling-laws"


def test_reprocessing_never_double_counts(session):
    seed(session)
    weight_before = session.conn.execute(
        "SELECT weight FROM aggregates"
    ).fetchone()[0]

    rebuild_tool(session, "all")
    rebuild_tool(session, "all")

    assert session.conn.execute("SELECT weight FROM aggregates").fetchone()[0] == (
        weight_before
    )


def test_an_external_edit_is_healed_on_reopen(session, tmp_path):
    seed(session)
    root = session.paths.root
    orphan = Capture(
        id="c_01HQZZZZZZ",
        created="2026-08-09T09:00:00Z",
        source="manual",
        why=None,
        text="Written straight into the vault by another tool.",
    )
    (root / "captures" / "2026-08-09-0900-ZZZZZZ.md").write_text(
        "---\nid: c_01HQZZZZZZ\ncreated: '2026-08-09T09:00:00Z'\nsource: manual\n---\n\n"
        + orphan.text
        + "\n"
    )
    session.close()

    with Session(root, embedder=StubEmbedder()) as reopened:
        found = reopened.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE doc_id = ?", ("c_01HQZZZZZZ",)
        ).fetchone()[0]
        assert found == 1


def test_a_malformed_file_does_not_disable_the_vault(session):
    seed(session)
    (session.paths.notes / "broken.md").write_text("---\nnot: [closed\n")
    session.resync()

    assert search_local(session, "data exhaustion")["hits"]
    assert any(
        issue["kind"] == "malformed_note" for issue in review_queue(session)["vault_issues"]
    )


def test_graph_stats_answers_is_global_search_worth_trying(session):
    seed(session)
    stats = graph_stats(session)
    assert stats["clustering"]["active"] is False
    assert stats["clustering"]["remaining"] > 0
    assert search_global(session, "themes?")["available"] is False
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_integration.py -v`
Expected: FAIL — the module imports fine but assertions fail until every prior
task is complete. If prior tasks all passed, expect PASS here.

- [ ] **Step 7: Run the entire suite**

Run: `uv run pytest -v`
Expected: PASS (all tests, one deselected by the `network` marker)

- [ ] **Step 8: Verify the real embedder path once, by hand**

Run: `uv run pytest -m network -v`
Expected: PASS (1 test; downloads ~90MB on first run)

`fastembed` is a default dependency, so a plain `uv sync` is enough — the shipped
`MINDPALACE.md` selects `kind: local`, and an optional dependency would mean the
default config could not start the default server.

- [ ] **Step 9: Smoke-test the CLI**

```bash
uv run mindpalace --vault /tmp/mp-smoke --init --check && echo "vault ok"
```
Expected: prints `vault ok`

- [ ] **Step 10: Commit**

```bash
git add mindpalace/server.py tests/test_server.py tests/test_integration.py
git commit -m "feat: MCP server wiring, CLI, and end-to-end invariants"
```

---

## Appendix: spec coverage

| Spec section | Task(s) |
|---|---|
| §3 Durability model — three tiers | 4, 14, 17 |
| §4.1 Identifiers (`g_` avoids the `c_` collision) | 1 |
| §4.2 Source tier, ULID in capture filename | 2, 3, 4 |
| §4.3 Assertions vs. aggregates, symmetric/directed | 1, 9 |
| §4.4 Generated tier, `user:` block, related markers | 4, 14 |
| §4.5 Cache tier | 7, 10 |
| §5 Modules and the Phase-1 log seam | 5–17 |
| §6 `MINDPALACE.md` schema and validation | 5 |
| §6 `cluster_weight` as a live experiment | 12 |
| §7 Fifteen tools | 15, 16, 17 |
| §7.1 Guided payloads | 15 |
| §7.2 Review gate as convention, no actor param | 15, 16 |
| §7.3 Dismissal tombstones the assertion only | 15, 16 |
| §7.4 Community lineage by Jaccard | 12, 16 |
| §8.1 What is embedded and indexed | 10 |
| §8.2 RRF ordering, raw-signal evidence gate | 11 |
| §8.3 Global search, missing/stale reports returned | 13 |
| §8.4 Citations validated on write | 13, 16 |
| §8.5 Alias resolution, no identity merging | 16 |
| §9.1 Write-ahead ops and replay | 6, 15 |
| §9.2 `flock` and compare-and-swap (detection, not prevention) | 3, 15 |
| §9.3 Vault and process contract | 5, 17 |
| §9.4 Network boundary reported by `graph_stats` | 8, 16 |
| §10 Failure modes | 10, 14, 16, 17 |
| §11 Testing, including the derivability invariant | every task; 17 for the invariant |

Requirements the first draft claimed here but did not actually implement, now
covered:

| Requirement | Task | What was wrong |
|---|---|---|
| §5 embedder mismatch forces a rebuild | 7, 15 | never checked; a swapped model returned silently empty |
| §10 malformed files stay searchable | 10 | parsed-and-dropped, so they vanished from FTS |
| §4.5 `index.md` is a maintained catalog | 10 | scaffolded once, never updated |
| §7 `review_queue` inlines endpoint snippets | 16 | returned ids only, forcing extra `read` calls |
| Tier 2 staleness is provable | 14 | hashed ids, so an edited note never marked its page stale |

