# 0002 · One document for every format

- **Status**: implemented on `feat/document-ir`, 2026-09-04; deviations from the design listed at the end
- **Related**: [0001](0001-what-we-borrowed-from-utopia.md) for the fold constraint every
  durable thing obeys; PRD §6.1 (text units) and §7.1 (content handling by medium).
  Utopia's `utopia-ingest` crate is the reference for the format list and the chunk budget.

> The PRD promised text units "from day one" and the code never built them: a capture
> is one string, and a paper would be one 100k-character string. This record adds the
> layer, and one intermediate representation that every format, including a typed
> thought, passes through on the way to it.

## What is in scope

File formats, reached through a local path: PDF, DOCX, PPTX, XLSX, XLS, XLSB, ODS, CSV,
TSV, HTML, Markdown, plain text. Plain text is decoded with encoding detection. An
unrecognised format falls through to plain text rather than failing.

Not in scope, by decision: any connector (watched folder, RSS, WebDAV, Notion, share
sheet), URL fetching, OCR, audio, images. Those come later and will hand files to the
same entry point.

## The IR

```
Document
  title        str | None    from the file itself; never invented
  source       SourceInfo    name, mime, sha256, size, parser, parser_version
  metadata     dict          only what the file states: author, created, modified,
                             page_count, sheet_names, ...
  blocks       tuple[Block]

Block
  kind         heading | paragraph | list_item | table | code
  text         plain text; a table is a GFM table
  level        heading depth (1-6), else None
  locator      Locator | None

Locator
  kind         page | slide | sheet | section | line
  label        "3", "2", "Q1", "Introduction", "40"
  index        int, ordinal within its kind
```

Two consumers:

- `to_markdown(document)` renders the capture body. Locators become headings the way
  Utopia writes "## Page 3", so a reader of the file sees where each piece came from.
- `to_text_units(document)` packs whole blocks up to `UNIT_BUDGET` (1200 characters)
  with `UNIT_OVERLAP` (150) carried from the previous unit's tail. Only an oversized
  paragraph is split, and only at sentence ends. A table row is never split. A heading
  travels with the block that follows it. A capture whose body fits one unit has one
  unit, and the layer is invisible, as the PRD says.

`save_capture` builds a Document from the text parser, so a pasted article is chunked
the same way a PDF is. There is exactly one path.

## Parsers

Dispatch is by extension, then by sniffing (`%PDF`, a zip whose entries name `word/`,
`ppt/`, or `xl/`, an ODS `mimetype` entry), then plain text.

| Format | Library | Blocks and locators |
|---|---|---|
| PDF | pypdf | text layer per page; `page` locator; title/author/dates from the info dict |
| DOCX | stdlib zip + XML | `w:p` paragraphs; `Heading N` styles become headings; `w:tbl` becomes a table; `section` locator from the nearest heading |
| PPTX | stdlib zip + XML | one `slide` locator per `ppt/slides/slideN.xml` in numeric order; the title placeholder becomes a heading |
| XLSX, XLS, XLSB, ODS | python-calamine | one table block per sheet, `sheet` locator, capped at 2000 rows with the cap recorded in metadata |
| CSV, TSV | stdlib csv | one table block, same cap, delimiter sniffed for CSV |
| HTML | beautifulsoup4 | `<title>`; content root is `main`, `[role=main]`, or `article` when present, else body; `h1`-`h6` become headings, `p`/`li`/`pre`/`table` map to their kinds; scripts and styles dropped |
| Markdown | stdlib | ATX headings, fenced code, everything else paragraphs split on blank lines; body kept verbatim |
| Text | charset-normalizer | paragraphs split on blank lines; no locator, the unit offsets are enough |

A parser that raises produces a capture anyway: the plain-text fallback runs on the
bytes it can decode, and the capture's metadata records `parser_error`. Nothing is lost
silently; the ingest response says what happened.

## Storage

Source set (Tier 1):

- `captures/attachments/<sha256>.<ext>`: the original bytes, written once. The PRD's
  immutable raw payload. A better parser later can re-read it.
- The capture file: front-matter gains `attachment` (relative path), `sha256`, `mime`,
  `title`, `parser`, `parser_version`, `metadata`, and `units`, a list of
  `[start, end]` character offsets into the body. The body is the rendered markdown.

Unit offsets live in the source set so unit ids are stable: `u_<capture ulid>_<ordinal>`.
A future chunker change alters offsets only for captures ingested after it, never the
ids that notes already cite. The cache derives everything else.

Deduplication: a file whose sha256 matches an existing attachment returns that capture
and writes nothing. The response says so.

Cache (Tier 3): a `text_units` table (id, capture_id, ordinal, start, end, locator,
text). `docs` and `vectors` index each unit of a multi-unit capture as kind `unit`,
and the capture itself only when it has one unit, so a hit lands on a page or a section
and a long capture is not returned twice.

## Provenance

`write_note` accepts an optional `text_unit_ids` list on each entity, relationship, and
claim. Each id must exist and belong to the note's capture, or the item is dropped with
reason `unknown_text_unit` (0001 §2). The ids are stored in the note's front-matter,
carried by the fold, projected into the cache, and included in the evidence hashes.
The citation id pattern accepts the `u_` prefix so reports can cite a unit.

## Tool surface

- `ingest_file(path, why=None, source="file")`: parses, stores, chunks, syncs. Returns
  the capture id, title, attachment path, parser, unit count, the first few units
  (id, locator, first line), whether it was a duplicate, any parser error, and an
  extraction instruction that tells the assistant to read units by id and pass
  `text_unit_ids` when it writes the note.
- `read(u_...)` returns the unit text, its locator, ordinal, and capture id.
- `save_capture` is unchanged in signature; its response gains `units`.
- `graph_stats` gains `text_units` and `attachments` counts.

## Not decided here

How a file arrives without a person naming its path. That is the connector question
and it is explicitly deferred.

## Deviations found while implementing

- **Unit ids of a typed capture.** `save_capture` keeps the user's text verbatim
  rather than storing the IR's rendering, so its unit offsets are mapped back onto the
  original string. If the mapping cannot find a unit's text (it always can for text
  the parser itself produced), the capture falls back to a single unit rather than
  storing offsets that do not line up.
- **Locator of a plain-text unit.** The spec said "no locator, the unit offsets are
  enough". In the cache, a unit's locator is read off the nearest `## Page N`-style
  heading at or before it, so plain-text units simply have none; nothing was added.
- **Overlap is exact, even mid-word.** The chunker steps back exactly 150 characters
  into the previous unit for an oversized paragraph. A word-boundary overlap would
  have been prettier and less predictable; the overlap is a retrieval aid, not prose.
- **Nested HTML tables** render once, inside the outermost table, rather than three
  times as the reference implementation would have; found by the parser's implementer.
- **`Heading0` in DOCX** clamps to level 1 instead of crashing into the text fallback.
