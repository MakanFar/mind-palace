# SDD ledger — plan: docs/superpowers/plans/2026-08-08-mind-palace-mcp.md

Branch: feat/mcp-server
Repo initialised at c676fd6 (docs only); implementation starts from there.
17 tasks. Pre-flight scan run before Task 1.

Pre-flight: fixed vacuous assert (Task 9) and unused conftest fixture (Task 8) in plan; commit ce2bd0e.
BASE for Task 1: ce2bd0e
Task *: minor (deferred): six near-duplicate config() fixtures across Tasks 9-14 are deliberate (task isolation, and they differ in edge types) — final review to triage, task reviewers should not re-flag.
Task 1: complete (commits ce2bd0e..0d35abf, review clean)
Task 1: minor (deferred): uv.lock untracked (brief's git-add list excludes it); final review to triage whether to commit it.
Task 2: fix round 1/5 (2 addressed, 0 open — falsy non-mapping accepted; lstrip destroyed leading blank lines; commits 780b31b..f4e1356)
Task 2: complete (commits 0d35abf..f4e1356, review clean). Plan reference code corrected to match.
Task 3: minor (deferred): atomic_write leaves files mode 0600 (mkstemp default preserved across rename) — harmless single-user, note for final review.
Task 3: minor (deferred): atomic_write failure/cleanup path verified by reading, not by a test.
Task 3: fix round 1/5 (1 addressed, 0 open — cas_write(None) check-then-act replaced with os.link exclusive create; commits 4329cc5..5898ff0)
Task 3: minor (deferred): atomic.py cas_write has a redundant `except ConflictError` clause identical to the `except BaseException` below it.
Task 3: complete (commits c7847b9..5898ff0, review clean). Plan reference code corrected to match.
Task 4: adjudicated — "entity id derived from filename" finding REJECTED: spec §4.1 makes entities slug-identified by design; the controller's constraint text was wrong and has been corrected in the plan's Global Constraints.
Task 4: minor (deferred): store.py write_entity_page builds "e_" + page.slug inline instead of calling ids.entity_id(), and does not slugify page.slug — can diverge from entity_path(), which does normalize.
Task 4: minor (deferred): stale defaults inconsistently on hand-edited files (True for entity pages, False for reports).
Task 4: fix round 1/5 (1 addressed, 1 adjudicated-no-change; commits 631a318..a7f4ad1)
Task 4: minor (deferred): _split_related takes the LAST standalone marker line rather than the first (only reachable via manual external editing).
Task 4: complete (commits ce6f277..a7f4ad1, review clean)
PLAN-SYNC DEBT: plan reference code for vault/store.py (_split_related + write_entity_page validation) not yet synced to shipped code. Do one sync pass over all remaining drift before the final review.
PROCESS CHANGE (from Task 6): implementer prompts now grant explicit licence to fix genuine correctness defects in the brief's sample code (silent corruption, check-then-act, unvalidated coercion), flagging the deviation, while leaving exact values/names/signatures untouched. Rationale: Tasks 2-5 each burned a fix round on a defect in the plan's illustrative code rather than on implementer error.
Task 5: fix round 1/5 (5 addressed: HOME-guard resolve, nested unknown keys, directed bool, cluster_weight wrap, entity_types list; 1 open MINOR; commits e0fad3a..9bfad59)
Task 5: minor (deferred): non-empty-dir error still recommends "--init", which then fails with a different message.
Task 5: minor (deferred): test_open_vault_error_message_tailored_for_non_empty_dir_without_init is VACUOUS — asserts a substring present in the pre-fix message too. Final review should replace or delete it; a test that cannot fail is worse than none.
Task 5: minor (deferred): non-dict `thresholds`/`edge_types` scalars raise bare AttributeError/TypeError rather than ConfigError.
Task 5: complete (commits a7f4ad1..9bfad59, review clean on all blocking findings)
Task 6: CONTROLLER RULING (not escalated): _read_lines JSONDecodeError on truncated trailing line is plan-mandated, but the standing licence issued to implementers from Task 6 onward already authorises correcting such defects, and the human ruled "correctness governs" on five consecutive equivalent findings (Tasks 2,3,4,5). Re-asking would be ceremony. Fix dispatched directly; rule adopted: tolerate a malformed FINAL line (torn write), raise loudly on a malformed interior line (real corruption).
Task 6: minor (accepted, no fix): commit() accepts an op id never begun; _append_line has no cross-process lock (single-writer by design).
Task 6: fix round 1/5 (1 addressed — torn-final-line tolerated, interior corruption raises with file+line; commits ad65596..dda552e)
Task 6: complete (commits 9bfad59..dda552e, review clean)
Task 7: complete (commits dda552e..eafb95f, review clean, no fix round)
Task 7: minor (deferred): PRAGMA foreign_keys=ON but no table declares a FOREIGN KEY — inert config; whichever later task owns referential integrity should decide.
Task 7: minor (deferred): vector dimension-mismatch raises ValueError from np.vstack, untested; acceptable under Tier-3 delete-and-rebuild.
Task 7: minor (deferred): task-7-report.md mis-states that empty `kinds` yields invalid SQL — `IN ()` is valid SQLite returning zero rows. Report wrong, code fine.
Task 8: resolved ⚠️ from review — get_embedder takes config.embedder (a dict), not the Config dataclass; Task 15's session.py already calls get_embedder(self.config.embedder). Not a gap.
Task 8: minor (deferred): LocalEmbedder.__init__ wraps only ImportError; a bad model_name propagates fastembed's native exception rather than EmbedderError.
Task 8: fix round 1/5 (3 addressed — cloud guardrail test w/ message match, float32 dtype pin, zero-norm asymmetry comment; commits b50e023..567435d)
Task 8: complete (commits eafb95f..567435d, review clean)
Task 9: CONTROLLER DESIGN DECISION — fold now raises a typed hierarchy (FoldError base; DuplicateAssertionIdError{assertion_id,note_ids}; UnknownEdgeTypeError{edge_type,note_id,assertion_id}; UnknownDecisionActionError{action,assertion_id}) with STRUCTURED ATTRIBUTES, not message text. Rationale: sync (Task 10) calls fold once over the whole vault and must satisfy spec §10 "one bad file never renders the vault unusable" — it needs to identify and quarantine the offending note, then retry. CARRY THIS INTO TASK 10's BRIEF.
Task 9: also fixed — aggregate_shape no longer string-parses aggregate_key output (an edge type containing "|" silently corrupted source/type/target and therefore rank); DecisionLog.append now validates action at write time so a bad action can never be durably written.
Task 9: minor (deferred): fold does not validate EntityInstance.type against config.entity_types — a typo'd type is stored as-is.
Task 9: minor (deferred): duplicate Note objects with same (created,id) and no assertions resolve entity type by caller order; caller-bug-only.
Task 9: fix round 1/5 (4 addressed — typed FoldError hierarchy w/ structured attrs, direct aggregate tuple capture, mixed-status test, DecisionLog.append action validation; commits ac20d61..2874e95)
Task 9: complete (commits 567435d..2874e95, review clean). 128 tests green.
Task 10: complete (commits 2874e95..900dd49, review clean, no fix round). 146 tests green. Quarantine-and-retry around fold implemented and verified; implementer's removal of the redundant duplicate-id pre-check judged correct (x_/k_ id spaces are disjoint by construction).
Task 10: minor (deferred): unreachable note_paths.get fallback; sync.py at 391 lines could split out _fold_with_quarantine later; no 3-way-collision or mixed-FoldError-type test.
Task 11: PLAN DEFECT (human ruled: fix the tests, not the code). Two plan tests were unsatisfiable:
  (a) test_rrf_prefers_documents_appearing_in_both_lists — all three docs appear in BOTH lists; by convexity a=c=1/61+1/63 > b=2/62, so the asserted winner can never win at any k. Fixture never set up the property the name claims.
  (b) test_local_search_returns_lexical_matches — SQLite BM25 IDF = log((N-df+0.5)/(df+0.5)) is exactly 0 at N=2,df=1, so a real match scores 0 against the 2.0 floor; StubEmbedder's random vectors can't clear the cosine bar either. Fixture enlarged to ~10 docs rather than lowering the floor, so the production default stays exercised.
Task 11: STRUCTURAL NOTE for later tasks — StubEmbedder makes semantic relevance untestable by construction, so any test that must pass the evidence gate has to pass on BM25 alone, and BM25 is degenerate on tiny corpora. Size retrieval fixtures accordingly.
Task 11: deviations approved — abstain safety sentence is always appended (a custom MINDPALACE.md template must not be able to delete the anti-fabricated-provenance instruction); success path returns "note": None for uniform response shape.
Task 11: fix round 1/5 (2 plan-defective tests repaired, no production change; commits 63bb46e..1f52cb5)
Task 11: complete (commits 900dd49..1f52cb5, review clean). 157 tests green.
Task 11: minor (deferred): bm25 negative-score clamp to 0.0 is undocumented and untested; `k` names two unrelated concepts (RRF smoothing constant vs top-N) in one module.
Task 12: PLAN DEFECTS (2, both Important):
  (a) match_lineages greedy over `fresh` in Leiden's arbitrary cluster order can orphan an EXACT-membership continuation. Counterexample: P={c,d,e}; F1={c,d,e,x} J=0.75 claims P first; F2={c,d,e} J=1.0 then mints a new lineage. Fixed by scoring all pairs and assigning highest-first. (Standing policy; not escalated.)
  (b) Community.parent encoded "{level-1}:{cluster}" but Community never stored its own cluster index, so the pointer was unresolvable, and every test fixture was too small to produce a second level so the path was untested. HUMAN RULED: parent now holds the parent's LINEAGE ID, resolved in a second pass in partition and remapped through match_lineages' old->new map. No DB change (communities.parent is already TEXT).
Task 12: also fixing scoped filterwarnings for hyppo/numba deprecation noise from graspologic's import chain, so the suite runs pristine without a blanket ignore.
Task 12: fix round 1/5 (3 addressed — global highest-first lineage assignment, parent-as-lineage-id w/ remap, scoped filterwarnings; commits 9f655c6..6441bea)
Task 12: complete (commits 1f52cb5..6441bea, review clean). 170 tests green, output pristine.
Task 12: minor (deferred): the two multi-level hierarchy tests each rebuild the same 250k-edge gnp_random_graph and re-run hierarchical_leiden independently (~2.3s each, no fixture sharing); a module-scoped fixture would halve it, a dense clique just over max_cluster_size would cut it further.
Task 12: minor (deferred): task-12-report.md reports test_cluster.py alone at 15.37s but the full 170-test suite at 11.27s — almost certainly numba disk-cache warm-up, but the report leaves the contradiction unexplained.
Task 13: complete (commits 6441bea..4739d85, review clean, no fix round). 182 tests green.
Task 13: deviations approved & verified — RANK_TIEBREAK_WEIGHT 0.001 -> 0.000001 (at 0.001 a 0-10 impact rank contributed up to 0.01 against a max RRF relevance of ~0.033, so it could override real relevance rather than break ties); uniform response keys on both global_search branches.
Task 13: minor (deferred): citations ID_PATTERN can over-match a subscript-like token (e.g. n_1) inside a [Data: ...] block; retrieve.py uses two different idioms for reconstructing a ranked list from vector hits.
Task 14: implementer deviation VERIFIED CORRECT — entity_input_hash now hashes ALL assertions touching an entity, not only those in a traversable aggregate; the brief's version skipped unconfirmed aggregates before reaching assertion_ids, so editing a proposed assertion's rationale left the page pinned stale:false.
Task 14: HUMAN RULED — community_input_hash must match entity_input_hash's evidence breadth (member entity descriptions, all assertions touching a member regardless of status, all claims regardless of status). Rationale: cluster_tool passes only slugs/pairs/weights, but the assistant can read or get_entity any member before writing, so those are reachable evidence. Accepts more frequent staling; a stale flag is advisory and cheap, a falsely-fresh report is the failure the module exists to prevent.
Task 14: minor (deferred): write_entity_page/write_report use plain atomic_write, not cas_write (unlike write_note/write_capture) — a hand-edit racing rebuild's read-then-write can be lost. Pre-existing Task 4 interface. FINAL REVIEW SHOULD TRIAGE.
Task 14: minor (deferred): rebuild rewrites every entity page on every call even when nothing changed, so the gated second re-sync fires on nearly every non-empty rebuild; mark_stale_reports ignores `scope`.
!! CARRY INTO TASK 16 BRIEF: community_input_hash signature is now (members, tables, notes_by_id) — NOT (members, tables). The plan's Task 16 code calls it two-arg in cluster_tool and write_community_report; both call sites must be updated or they TypeError. mark_stale_reports(conn, store, tables) is UNCHANGED.
Task 14: fix round 1/5 (2 addressed — community_input_hash widened to full evidence breadth, entity_input_hash pin tests; commits 46694cf..9b43a99)
Task 14: complete (commits 4739d85..9b43a99, review clean). 199 tests green.
Task 14: minor (deferred): mark_stale_reports re-reads store.iter_notes() though rebuild() already built an equivalent notes_by_id — accepted cost of keeping its signature frozen.
Task 15: complete (commits 9b43a99..f71989e, review clean, no fix round). 225 tests green.
Task 15: deviations verified necessary — Session.open() released the flock on partial-open failure (regression test uses the deterministic cloud-EmbedderError path); MINDPALACE.md extraction_next template had a hard line-wrap mid-phrase so the brief's own assertion could never match (config._parse_templates does not normalise newlines).
!! CARRY INTO TASK 16 BRIEF: report_next and abstain templates in mindpalace/templates/MINDPALACE.md still contain internal hard line-wraps of the same kind that broke extraction_next. Any test asserting a phrase from those templates verbatim will fail.
Task 15: minor (deferred): unreachable else-branch in write_note slug derivation; _known_entities does N+1 page reads; write_note rescans all captures per call.
Task 16: CONTROLLER RULING (not escalated): alias collisions resolved silently to the alphabetically-first slug, violating spec §8.5 ("never silently resolved in favour of one"). Standing policy + explicit binding constraint; fix dispatched — detection as a vault_issue in sync, ToolError on ambiguous alias lookup in _resolve_slug, exact slug match still wins.
Task 16: deviations all judged right, incl. repairing test_reclustering_marks_an_affected_report_stale_immediately (write_note's own rebuild flipped the report stale before cluster_tool ran, so the brief's assertion was unreliable) and another degenerate 2-doc BM25 fixture.
Task 16: minor (deferred): tools.py is 808 lines and reads as four concatenated modules (write/read/lifecycle/cluster); split into tools/ package recommended before further surface is added. FINAL REVIEW SHOULD TRIAGE.
Task 16: minor (deferred): read() returns a generic "not found" for valid x_/k_/op_ ids rather than distinguishing "not a readable kind".
Task 16: fix round 1/5 (1 addressed — alias collisions now detected as vault_issue in sync and raise ToolError in _resolve_slug, exact-slug precedence preserved; commits 19111b1..69e2bf6) BUT introduced new Important: _ambiguous_alias_issues counts per alias-occurrence not per page, so one entity listing an alias twice (or two case variants) is reported as colliding with itself.
Task 16: fix round 2/5 dispatched — per-page alias dedupe before the claimant count.
Task 16: fix round 2/5 (1 addressed — per-page alias dedupe before claimant count; commits 69e2bf6..924f534)
Task 16: complete (commits f71989e..924f534, review clean). 267 tests green.
Task 17: deviations all verified right — installed mcp is 2.0.0 and has no fastmcp submodule (adapted to MCPServer); seed() fixture was another degenerate 4-doc BM25 corpus; derivability snapshot strengthened to observe entity-page related wikilinks, which the brief's version deleted but never checked.
Task 17: fix round 1/5 dispatched — tighten mcp pin to >=2.0.0 (the declared >=1.2.0 permits versions where server.py's import fails), plus at least one genuine round trip through the MCPServer object (test_server.py only ever called list_tools(), so an argument-order bug in any of the 15 delegations would pass every test).
Task 17: fix round 1/5 (2 addressed — mcp pin >=2.0.0, real MCPServer.call_tool round-trip tests; commits 95f9f08..838145c). The round-trip tests SURFACED a latent defect that 280 tests missed: anyio.to_thread.run_sync runs sync tool bodies on worker threads and sqlite3's check_same_thread=True rejected the shared connection, so EVERY tool call would have crashed under the real transport.
Task 17: fix round 2/5 dispatched — check_same_thread=False alone is insufficient. Re-reviewer read mcp==2.0.0's dispatcher: tools/call is spawned fire-and-forget (only "initialize" is in inline_methods), so concurrent tool calls genuinely share one connection and one implicit transaction. Most write tools are framed only by session.operation() (oplog bookkeeping), not by `with session.conn:`, so a concurrent read can observe a half-written state or commit another thread's in-flight operation early. Fix: threading.RLock on Session, acquired via a single _run helper in server.py's delegation layer (the only transport entry point), plus correcting db.py's false comment about the transport serialising requests.
Task 17: fix round 2/5 (1 addressed — session.lock RLock + single _run helper routing all 15 tools + corrected db.py comment; commits 838145c..01c00c4). Re-reviewer independently verified non-vacuously in an isolated scratch copy: lock disabled -> concurrency test FAILED with "UNIQUE constraint failed: files.path", lock-held test FAILED; restored -> both pass.
Task 17: complete (commits 924f534..01c00c4, review clean). 285 tests green.
ALL 17 TASKS COMPLETE. Proceeding to final whole-branch review.

=== FINAL WHOLE-BRANCH REVIEW (opus, c676fd6..01c00c4) ===
Verdict: With fixes. Found 2 Critical + 8 Important that no per-task review could see.
CRITICAL 1: fold() called BARE at rebuild.py:203 and tools.py:264 (_tables), defeating sync's quarantine entirely — one malformed note leaves search working but every write tool raising. Task 10 built the quarantine; Tasks 14/16 called around it.
CRITICAL 2: Tier-2 reads unguarded — read_entity_page does data["type"], read_report indexes 4 keys, iter_* wrap nothing; one malformed entity page kills resync and every write path. Tier 2 is the tier users are told to hand-edit.
IMPORTANT 3: _split_related discards everything after the close marker — a hand-written section below the related block is destroyed by the next rebuild (which runs on every capture).
IMPORTANT 4: cluster_tool's needs_report compares input_hash AFTER mark_stale_reports overwrote it, so it reports needs_report:False for the report it just staled; the prose-refresh loop never fires.
HUMAN RULED fix-wave scope: reviewer's merge gate + corrupt-SQLite recovery + entities//communities/ added to drift set. DEFERRED with spec amendment: read-path self-heal call sites, content-hash-keyed vectors (write-once embeddings).
FINAL FIX WAVE: all 9 findings ADDRESSED (re-review mutation-tested each; 8/9 fully discriminate). 314 tests.
FINAL FIX WAVE introduced 1 CRITICAL regression: sync.py:317 page.user.get() unguarded + read_entity_page accepting any YAML for `user`; finding 8's drift-set addition promotes it to an open-time AttributeError, so a scalar `user:` (the channel the page banner instructs users to use) makes the server refuse to start. Delta confirmed: 01c00c4 opened, 1c89ef6 did not. HUMAN RULED: fix it (validate user is a mapping -> FrontMatterError -> existing degrade path), plus same guard in _resolve_slug, plus repair 2 tests that mutation testing showed no longer discriminate.
PARKED as follow-ups (with rulings): iter_notes/iter_captures residual (write_note raises on a malformed capture; read() order-dependent; vault still self-diagnoses via review_queue and all other tools work — NOT write-dead); write_community_report overwrites unconditionally; duplicated related-block makes writes raise; triple note parse per rebuild; write_entity_description can emit raw ConflictError; hand-added entities/My Entity.md silently skipped; user.aliases-as-string iterates per character.
DEFERRED with spec amendment needed: read-path self-heal in read/local_search (spec §10 promises it), content-hash-keyed vectors (spec §8.1 designs capture/note vectors as write-once).
