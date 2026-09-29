# 0005 · A human confirms

- **Status**: accepted, not yet implemented, 2026-09-29
- **Related**: PRD §5 P7 (durable structure is review-gated) and §11 (the hallucination
  feedback loop, the #1 risk); [0001](0001-what-we-borrowed-from-utopia.md) §1 and §5 (adoptions
  and merges are logged decisions); [0003](0003-a-window-into-the-palace.md) §Part 2 (the
  window writes `decisions.jsonl`); [0004](0004-retiring-an-entity.md) (retirement).

> The review gate is only as strong as whoever may pass through it. Today that is anyone
> holding the MCP connection: the assistant that proposed a relationship can call
> `resolve_assertion(confirm)` on it in the next breath, and the only thing in the way is
> a docstring asking it not to. The same holds for `adopt_type`, `merge_entities`, and
> `retire_entity`. The log cannot tell afterwards, because every such line says
> `via: "resolve_assertion"` whoever decided. This record makes structure-adding
> decisions reachable only by a person: through a prompt the server puts to the user
> directly (MCP elicitation), through the Obsidian window, or through a new CLI.

## Decisions taken in brainstorming

| Question | Decision |
|---|---|
| Where can a human confirm | Chat, Obsidian, and a CLI. Chat only through a server-driven review the user opts into. |
| Who asks "review now?" | The server, by elicitation. A question the assistant asks in its own text can be answered by the assistant. |
| What the assistant may still do alone | Propose, dismiss, reopen. Dismissal removes structure; it never adds any. |
| `adopt_type`, `merge_entities`, `retire_entity` | Same gate, every action including `revoke`, `unmerge`, `restore`: one elicited approval each, or the CLI (or the window, for retire). |
| A client that cannot elicit | Refused, never trusted. Confirmation then waits for the window or the CLI. |
| Mechanism | Elicit inside the tool call. A staged-decisions log was considered and set aside: it needs a new log, fold rules, and plugin UI to serve "approve later", which the window and CLI already serve. |

## Part 1: the chat review

A new MCP tool, `review(limit=20)`:

1. If the client did not declare the elicitation capability, raise: the client cannot ask
   the user directly; confirm in Obsidian or with `mindpalace review`.
2. Read the proposed relationships and claims, in `review_queue`'s order (relationships
   by strength, then claims). None: return "nothing to review" without eliciting.
3. Elicit "N proposals are waiting (R relationships, C claims). Review them now?". Decline
   or cancel returns `{reviewed: false}` with a `next` saying confirmation now waits for
   Obsidian or the CLI, and the assistant may still dismiss or reopen.
4. Accept: elicit once per proposal. The message carries what `review_queue` carries: for
   a relationship the pair, arrow and type, rationale, strength, direction-correction
   flag, both endpoint snippets, and the note; for a claim the subject, text, validity,
   and the id it supersedes. The response schema is `decision: confirm | dismiss | skip |
   stop` and an optional `reason`. Cancel is `stop`.
5. Each decision is appended as it is made, under `session.operation`, with
   `via: "chat_review"`, after checking under the lock that the item is still `proposed`.
   One that is not (decided in the window meanwhile) is recorded in `already_decided` and
   not written.
6. One rebuild at the end. A session that dies midway loses nothing: `decisions.jsonl` is
   in the drift set, so the next open re-derives from it.
7. Return counts confirmed, dismissed, skipped; `already_decided`; whether the user
   stopped early. The assistant sees the outcome, never supplies it.

## Part 2: the single-action gate

`resolve_assertion` takes `dismiss` and `reopen` only. `confirm` raises, naming `review`,
the window, and the CLI.

`adopt_type`, `merge_entities`, `retire_entity`, every action:

1. **Validate under the lock.** Every existing refusal (unknown slug, merge cycle, live
   blockers, a new edge type without `directed`, unknown domain types) happens here, so
   the user is never asked to approve something that would be refused.
2. **Elicit with the lock released.** The message states the concrete effect: "Merge `a`
   into `b`: `a`'s 3 assertions and 1 claim fold into `b`. Reason: …"; "Adopt wording
   'inspires' as edge type `supports` (existing): retypes 4 assertions"; "Adopt
   'authored' as a new edge type (directed, domain [person], range [paper]); adds it to
   MINDPALACE.md"; "Retire `foo`. Reason: …". Accept allows; decline or cancel raises
   "declined by the user; nothing was written".
3. **Re-validate under the lock, then write.** MINDPALACE.md is edited only here. A
   refusal that appears between ask and write (a cycle, a new live item) is raised as
   usual.

The lock is released while waiting because `_run` holds `session.lock` for the whole
dispatch, and a pending human answer would otherwise block every other tool call. The
gated handlers become `async`, since elicitation is; the rest stay synchronous.

The elicitation helper and the lock choreography live in `server.py`. `tools.py` stays
free of MCP types: it gains `decide(session, identifier, action, via, reason)` (the one
function that appends a review decision, used by the chat review, the CLI, and tests),
and a keyword `via` on `adopt_type`, `merge_entities`, `retire_entity` that reaches the
log line.

## Part 3: what `via` means

`via` names who decided.

| `via` | Meaning |
|---|---|
| `chat_review` | The user, in a `review` session |
| `chat_approval` | The user, answering one elicited adopt, merge, or retire |
| `cli` | The user, at the `mindpalace` command |
| `obsidian` | The user, in the window (unchanged) |
| `resolve_assertion` | The assistant: dismiss and reopen only, from now on |

Old lines keep their `via`, and the fold never reads it, so nothing breaks. A `confirm`
with `via: "resolve_assertion"` can only predate this record.

## Part 4: the CLI

`mindpalace` gains subcommands; the bare `mindpalace --vault X` form keeps starting the
server so existing MCP configs are untouched.

- `mindpalace serve --vault X`: the server.
- `mindpalace review --vault X [--limit N]`: the Part 1 walk in the terminal, one
  proposal at a time, `c`/`d`/`s`/`q` plus an optional reason, `via: "cli"`.
- `mindpalace adopt|merge|retire …`: the tools' arguments; prints the Part 2 summary and
  asks `Proceed? [y/N]`, `via: "cli"`.

Every interactive command refuses when stdin is not a TTY: an agent with a shell could
otherwise pipe `y` into it. The vault's `flock` is taken per write, not per session, so
the CLI runs beside a live server.

## Trust assumptions

- The MCP client shows elicitations to a person. Claude Code does; the specification
  permits an agent-like client to answer them itself. A client that does is outside what
  this record can defend.
- Anything with write access to the vault's files can still append to `decisions.jsonl`
  or the other logs. This record closes the MCP surface, not the filesystem.

## Testing

- `tools.py`: `via` reaches each log; `resolve_assertion` refuses `confirm`; `decide`
  writes every action.
- `server.py`, with a fake elicitation responder scripted to accept, decline, or cancel:
  a review session through confirm, dismiss, skip, stop; "no" to the opening prompt; an
  item decided elsewhere mid-prompt; a client without elicitation; each gated tool
  accepted and declined; a merge refused on re-validation because a cycle appeared while
  the prompt was open.
- CLI: scripted stdin for the review and the `y/N` prompts; refusal on a non-TTY stdin.
- The decisions conformance fixture gains `chat_review` and `cli` lines, so the plugin's
  fold is shown to accept them.
- Existing tests that confirm through `resolve_assertion` move to `tools.decide`.

## Not in this version

Showing the cited text units in each review prompt (review finding #5). Approving
adopt, merge, or retire from the window. Any cryptographic proof that a human answered.
