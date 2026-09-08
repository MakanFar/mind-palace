/** The side panel (docs/decisions/0003 §Part 4). */

import type { Selection } from "./canvas";
import type { Action } from "./decisions";
import { type GraphClaim, type GraphData, type GraphEdge, type GraphEntity, type GraphUntyped, isLive, isLiveEdge, isolatedEntities } from "./graph";
import { retireBlockers } from "./retirements";
import { colourFor } from "./palette";
import type { QueueItem } from "./review";

/** The review queue and where the selection sits in it (-1: not in it). */
export interface ReviewContext {
  queue: QueueItem[];
  index: number;
  /** Review mode: the queue replaces the empty state. */
  listing: boolean;
  /** The queue is ordered by note, so the list shows note headings. */
  grouped: boolean;
}

const KEYS = "c confirm · d dismiss · n next · p previous · r reason";

export interface PanelActions {
  openFile(path: string): void;
  select(selection: Selection): void;
  /** One reason, one append, for every assertion given. */
  decide(assertions: string[], action: Action, reason: string | null): Promise<void>;
  retire(slug: string, reason: string | null): Promise<void>;
}

export function describeValidity(from: string | null, to: string | null): string {
  if (from && to) return `${from} → ${to}`;
  if (from) return `from ${from}`;
  if (to) return `until ${to}`;
  return "";
}

function atOrBefore(left: string, right: string): boolean {
  const n = Math.min(left.length, right.length);
  return left.slice(0, n) <= right.slice(0, n);
}

/** Same rule as the Python `_holds_at`: bounds compared on their common prefix. */
export function holdsAt(from: string | null, to: string | null, asOf: string): boolean {
  if (from !== null && !atOrBefore(from, asOf)) return false;
  if (to === null || to === "unknown") return true;
  return atOrBefore(asOf, to);
}

type Resolved =
  | { kind: "entity"; entity: GraphEntity }
  | { kind: "edge"; edge: GraphEdge }
  | { kind: "untyped"; item: GraphUntyped }
  | null;

/** The record a selection names in this graph, or null once a merge, an
 *  adoption or a re-sync has retired it. */
export function resolveSelection(selection: Selection, graph: GraphData): Resolved {
  if (selection.kind === "entity") {
    const entity = graph.entities.find((e) => e.slug === selection.slug);
    return entity ? { kind: "entity", entity } : null;
  }
  if (selection.kind === "edge") {
    const edge = graph.edges.find((e) => e.key === selection.key);
    return edge ? { kind: "edge", edge } : null;
  }
  const item = graph.untyped.find((u) => u.id === selection.id);
  return item ? { kind: "untyped", item } : null;
}

const GONE: Record<Selection["kind"], string> = {
  entity: "That entity is not in the graph.",
  edge: "That edge is no longer in the graph (a merge or a re-sync changed it).",
  untyped: "That assertion is no longer untyped, or is no longer in the graph.",
};

export class Panel {
  constructor(
    private readonly container: HTMLElement,
    private readonly actions: PanelActions,
  ) {
    container.classList.add("mp-panel");
  }

  /** The reason typed for an assertion, for a decision made from the keyboard. */
  reasonFor(assertion: string): string | null {
    const input = this.container.querySelector<HTMLInputElement>(`.mp-reason[data-assertion="${assertion}"]`);
    return input?.value.trim() || null;
  }

  focusReason(): boolean {
    const input = this.container.querySelector<HTMLInputElement>(".mp-reason");
    input?.focus();
    return input !== null;
  }

  show(selection: Selection | null, graph: GraphData, asOf: string | null, review: ReviewContext | null): void {
    // graph.json is polled and the panel re-rendered on every change; a
    // reason half-typed into a card must survive that, so carry the inputs'
    // text across by assertion id.
    const drafts = new Map<string, string>();
    for (const input of this.container.querySelectorAll<HTMLInputElement>(".mp-reason[data-assertion]")) {
      if (input.value) drafts.set(input.dataset.assertion!, input.value);
    }
    this.container.empty();
    if (!selection) {
      if (review?.listing) this.renderQueue(review.queue, review.grouped);
      else this.container.createEl("p", { cls: "mp-muted", text: "Select a node or an edge." });
      this.renderIsolated(graph);
      this.renderVocabulary(graph);
      return;
    }
    if (review && review.index >= 0) {
      this.container.createEl("p", {
        cls: "mp-muted mp-position",
        text: `${review.index + 1} of ${review.queue.length} to review · ${KEYS}`,
      });
    }
    const resolved = resolveSelection(selection, graph);
    if (!resolved) this.container.createEl("p", { cls: "mp-muted", text: GONE[selection.kind] });
    else if (resolved.kind === "entity") this.renderEntity(resolved.entity, graph, asOf);
    else if (resolved.kind === "edge") this.renderEdge(resolved.edge, graph);
    else this.renderUntyped(resolved.item, graph);
    for (const input of this.container.querySelectorAll<HTMLInputElement>(".mp-reason[data-assertion]")) {
      const draft = drafts.get(input.dataset.assertion!);
      if (draft) input.value = draft;
    }
  }

  // ---- entity ----------------------------------------------------------

  private renderEntity(entity: GraphEntity, graph: GraphData, asOf: string | null): void {
    const head = this.container.createDiv({ cls: "mp-head" });
    const dot = head.createSpan({ cls: "mp-dot" });
    dot.style.background = colourFor(entity.type);
    head.createEl("h3", { text: entity.slug });
    const meta = [entity.type, `rank ${entity.rank}`];
    if (entity.merged_from.length) meta.push(`also: ${entity.merged_from.join(", ")}`);
    this.container.createEl("p", { cls: "mp-muted", text: meta.join(" · ") });
    if (entity.description) this.container.createEl("p", { text: entity.description });
    const open = this.container.createEl("a", { cls: "mp-link", text: "Open page" });
    open.addEventListener("click", () => this.actions.openFile(`entities/${entity.slug}.md`));
    this.retireRow(entity.slug, graph);

    this.renderClaims(graph.claims.filter((c) => c.subject === entity.slug), asOf);
    this.renderEdgeList(entity.slug, graph);
    this.renderUnits(entity.text_unit_ids, graph);
  }

  private renderClaims(claims: GraphClaim[], asOf: string | null): void {
    if (!claims.length) return;
    const visible = asOf ? claims.filter((c) => holdsAt(c.valid_from, c.valid_to, asOf)) : claims;
    const hidden = claims.length - visible.length;
    this.section(`Claims (${claims.length})${hidden ? `, ${hidden} outside ${asOf}` : ""}`);
    const list = this.container.createEl("ul", { cls: "mp-list" });
    for (const claim of visible) {
      const item = list.createEl("li");
      const text = item.createSpan({ text: claim.text });
      if (claim.status === "superseded") text.addClass("mp-struck");
      const bits = [claim.status, describeValidity(claim.valid_from, claim.valid_to)].filter(Boolean);
      item.createSpan({ cls: "mp-muted", text: ` ${bits.join(" · ")}` });
      if (claim.status === "proposed") this.decisionButtons(item, claim.id);
    }
  }

  private renderEdgeList(slug: string, graph: GraphData): void {
    const touching = graph.edges.filter((e) => e.source === slug || e.target === slug);
    const groups: Record<string, GraphEdge[]> = { confirmed: [], proposed: [] };
    for (const edge of touching) {
      if (!isLiveEdge(edge)) continue;
      groups[edge.traversable ? "confirmed" : "proposed"].push(edge);
    }
    const untyped = graph.untyped.filter((u) => (u.source === slug || u.target === slug) && isLive(u));
    for (const [label, edges] of Object.entries(groups)) {
      if (!edges.length) continue;
      this.section(`${label[0].toUpperCase()}${label.slice(1)} (${edges.length})`);
      const list = this.container.createEl("ul", { cls: "mp-list" });
      for (const edge of edges) {
        const other = edge.source === slug ? edge.target : edge.source;
        const arrow = edge.directed ? (edge.source === slug ? "→" : "←") : "—";
        const row = list.createEl("li", { cls: "mp-row" });
        row.createSpan({ text: `${edge.type} ${arrow} ${other}` });
        row.addEventListener("click", () => this.actions.select({ kind: "edge", key: edge.key }));
      }
    }
    if (untyped.length) {
      this.section(`Untyped (${untyped.length})`);
      const list = this.container.createEl("ul", { cls: "mp-list" });
      for (const item of untyped) {
        const other = item.source === slug ? item.target : item.source;
        const row = list.createEl("li", { cls: "mp-row" });
        row.createSpan({ text: `“${item.proposed_type ?? "?"}” — ${other}` });
        row.addEventListener("click", () => this.actions.select({ kind: "untyped", id: item.id }));
      }
    }
  }

  private renderUnits(unitIds: string[], graph: GraphData): void {
    if (!unitIds.length) return;
    this.section(`Units (${unitIds.length})`);
    const list = this.container.createEl("ul", { cls: "mp-list" });
    for (const id of unitIds) {
      const unit = graph.units[id];
      const capture = unit ? graph.captures[unit.capture_id] : undefined;
      const label = `${unit?.locator ?? "unit"} · ${capture?.title ?? unit?.capture_id ?? id}`;
      const row = list.createEl("li", { cls: "mp-row", text: label });
      if (capture?.path) row.addEventListener("click", () => this.actions.openFile(capture.path));
    }
  }

  // ---- edge and untyped ------------------------------------------------

  private renderEdge(edge: GraphEdge, graph: GraphData): void {
    this.container.createEl("h3", { text: `${edge.source} ${edge.directed ? "→" : "—"} ${edge.target}` });
    this.container.createEl("p", {
      cls: "mp-muted",
      text: `${edge.type} · ${edge.traversable ? "confirmed" : "proposed"} · weight ${edge.weight}`,
    });
    this.section(`Assertions (${edge.assertions.length})`);
    const pending = edge.assertions.filter((a) => a.status === "proposed").map((a) => a.id);
    // With one pending card the card's own buttons are the same thing.
    if (pending.length > 1) {
      const all = this.container.createDiv({ cls: "mp-card mp-all" });
      all.createEl("p", { cls: "mp-muted", text: `${pending.length} proposed on this edge` });
      this.decisionButtons(all, pending, `all ${pending.length}`);
    }
    for (const assertion of edge.assertions) {
      const card = this.container.createDiv({ cls: "mp-card" });
      card.createEl("p", { text: assertion.description || "(no rationale)" });
      const bits = [assertion.status, `strength ${assertion.strength}`];
      if (assertion.direction_corrected) bits.push("direction corrected");
      card.createEl("p", { cls: "mp-muted", text: bits.join(" · ") });
      this.noteLink(card, assertion.note_id, graph);
      if (assertion.status === "proposed") this.decisionButtons(card, assertion.id);
    }
  }

  private renderUntyped(item: GraphUntyped, graph: GraphData): void {
    this.container.createEl("h3", { text: `${item.source} — ${item.target}` });
    this.container.createEl("p", { cls: "mp-muted", text: `untyped · wording “${item.proposed_type ?? "?"}” · ${item.status}` });
    const card = this.container.createDiv({ cls: "mp-card" });
    card.createEl("p", { text: item.description || "(no rationale)" });
    this.noteLink(card, item.note_id, graph);
    if (item.status === "proposed") this.decisionButtons(card, item.id);
    this.container.createEl("p", {
      cls: "mp-muted",
      text: "To give this wording a type, ask the assistant to run adopt_type.",
    });
  }

  private renderQueue(queue: QueueItem[], grouped: boolean): void {
    if (!queue.length) {
      this.container.createEl("p", { cls: "mp-muted", text: "Nothing left to review." });
      return;
    }
    this.section(`Review queue (${queue.length})`);
    this.container.createEl("p", { cls: "mp-muted", text: KEYS });
    let list: HTMLElement | null = null;
    let group: string | null = null;
    for (const item of queue) {
      // A heading per note when the queue is grouped by note: consecutive
      // items from one note share it, so this needs no knowledge of the order.
      const heading = item.notePath ?? item.note;
      if (grouped && heading !== group) {
        group = heading;
        this.container.createEl("p", { cls: "mp-muted mp-group", text: heading });
        list = null;
      }
      list ??= this.container.createEl("ul", { cls: "mp-list" });
      const row = list.createEl("li", { cls: "mp-row" });
      row.createSpan({ text: item.label });
      if (item.pending.length > 1) row.createSpan({ cls: "mp-muted", text: ` ×${item.pending.length}` });
      row.addEventListener("click", () => this.actions.select(item.selection));
    }
  }

  /** Entities with nothing live attached (docs/decisions/0004): the ones
   *  retire would accept, listed so they can be walked one by one. */
  private renderIsolated(graph: GraphData): void {
    const isolated = isolatedEntities(graph);
    if (!isolated.length) return;
    this.section(`Isolated (${isolated.length})`);
    this.container.createEl("p", { cls: "mp-muted", text: "Nothing live names these. Select one to retire it (x)." });
    const list = this.container.createEl("ul", { cls: "mp-list" });
    for (const entity of isolated) {
      const row = list.createEl("li", { cls: "mp-row" });
      const dot = row.createSpan({ cls: "mp-dot" });
      dot.style.background = colourFor(entity.type);
      row.createSpan({ text: ` ${entity.slug}` });
      row.addEventListener("click", () => this.actions.select({ kind: "entity", slug: entity.slug }));
    }
  }

  private retireRow(slug: string, graph: GraphData): void {
    const blockers = retireBlockers(slug, graph);
    const row = this.container.createDiv({ cls: "mp-actions" });
    const reason = row.createEl("input", {
      cls: "mp-reason",
      attr: { placeholder: "reason (optional)", "data-retire": slug },
    });
    reason.addEventListener("keydown", (event) => {
      if (event.key === "Escape") reason.blur();
    });
    const retire = row.createEl("button", { cls: "mp-retire", text: "Retire" });
    if (blockers) {
      retire.disabled = true;
      retire.title = blockers;
      this.container.createEl("p", { cls: "mp-muted", text: `Cannot retire: ${blockers}.` });
      return;
    }
    retire.addEventListener("click", async () => {
      retire.disabled = true;
      try {
        await this.actions.retire(slug, reason.value.trim() || null);
      } finally {
        retire.disabled = false;
      }
    });
  }

  /** The reason typed for retiring an entity, for the keyboard. */
  retireReasonFor(slug: string): string | null {
    const input = this.container.querySelector<HTMLInputElement>(`.mp-reason[data-retire="${slug}"]`);
    return input?.value.trim() || null;
  }

  private renderVocabulary(graph: GraphData): void {
    if (!graph.vocabulary.length) return;
    this.section(`Vocabulary proposals (${graph.vocabulary.length})`);
    const list = this.container.createEl("ul", { cls: "mp-list" });
    for (const v of graph.vocabulary) {
      list.createEl("li", { text: `${v.kind}: “${v.proposed}” ×${v.count} — ${v.example}` });
    }
  }

  // ---- bits ------------------------------------------------------------

  private section(title: string): void {
    this.container.createEl("h4", { cls: "mp-section", text: title });
  }

  private noteLink(parent: HTMLElement, noteId: string, graph: GraphData): void {
    const path = graph.notes[noteId]?.path;
    if (!path) return;
    const link = parent.createEl("a", { cls: "mp-link", text: "Open note" });
    link.addEventListener("click", () => this.actions.openFile(path));
  }

  private decisionButtons(parent: HTMLElement, assertions: string | string[], suffix = ""): void {
    const ids = typeof assertions === "string" ? [assertions] : assertions;
    const row = parent.createDiv({ cls: "mp-actions" });
    const reason = row.createEl("input", {
      cls: "mp-reason",
      // A batch row's draft survives re-renders under its joined ids.
      attr: { placeholder: "reason (optional)", "data-assertion": ids.join(" ") },
    });
    // Escape hands the keys back to the view: while the field has focus,
    // c and d type letters rather than decide.
    reason.addEventListener("keydown", (event) => {
      if (event.key === "Escape") reason.blur();
    });
    const label = (verb: string) => (suffix ? `${verb} ${suffix}` : verb);
    const confirm = row.createEl("button", { cls: "mp-confirm", text: label("Confirm") });
    const dismiss = row.createEl("button", { cls: "mp-dismiss", text: label("Dismiss") });
    const act = (action: Action) => async () => {
      confirm.disabled = dismiss.disabled = true;
      try {
        await this.actions.decide(ids, action, reason.value.trim() || null);
      } finally {
        // A successful decide re-renders the panel and replaces these
        // buttons; a failed one must hand them back.
        confirm.disabled = dismiss.disabled = false;
      }
    };
    confirm.addEventListener("click", act("confirm"));
    dismiss.addEventListener("click", act("dismiss"));
  }
}
