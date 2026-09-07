/** The side panel (docs/decisions/0003 §Part 4). */

import type { Selection } from "./canvas";
import type { Action } from "./decisions";
import type { GraphClaim, GraphData, GraphEdge, GraphEntity, GraphUntyped } from "./graph";
import { colourFor } from "./palette";

export interface PanelActions {
  openFile(path: string): void;
  select(selection: Selection): void;
  decide(assertion: string, action: Action, reason: string | null): Promise<void>;
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

export class Panel {
  constructor(
    private readonly container: HTMLElement,
    private readonly actions: PanelActions,
  ) {
    container.classList.add("mp-panel");
  }

  show(selection: Selection | null, graph: GraphData, asOf: string | null): void {
    this.container.empty();
    if (!selection) {
      this.container.createEl("p", { cls: "mp-muted", text: "Select a node or an edge." });
      this.renderVocabulary(graph);
      return;
    }
    if (selection.kind === "entity") {
      const entity = graph.entities.find((e) => e.slug === selection.slug);
      if (entity) this.renderEntity(entity, graph, asOf);
      else this.container.createEl("p", { cls: "mp-muted", text: `${selection.slug} is not in the graph.` });
    } else if (selection.kind === "edge") {
      const edge = graph.edges.find((e) => e.key === selection.key);
      if (edge) this.renderEdge(edge, graph);
      else this.gone("That edge is no longer in the graph (a merge or a re-sync changed it).");
    } else {
      const item = graph.untyped.find((u) => u.id === selection.id);
      if (item) this.renderUntyped(item, graph);
      else this.gone("That assertion is no longer untyped, or is no longer in the graph.");
    }
  }

  /** A fingerprint of everything the panel would render for a selection, so
   *  a poll that changed nothing relevant does not wipe a half-typed reason. */
  static snapshot(selection: Selection | null, graph: GraphData, asOf: string | null): string {
    if (!selection) return JSON.stringify(["none", graph.vocabulary]);
    if (selection.kind === "entity") {
      const slug = selection.slug;
      return JSON.stringify([
        asOf,
        graph.entities.find((e) => e.slug === slug) ?? null,
        graph.claims.filter((c) => c.subject === slug),
        graph.edges.filter((e) => e.source === slug || e.target === slug),
        graph.untyped.filter((u) => u.source === slug || u.target === slug),
      ]);
    }
    if (selection.kind === "edge") return JSON.stringify(graph.edges.find((e) => e.key === selection.key) ?? null);
    return JSON.stringify(graph.untyped.find((u) => u.id === selection.id) ?? null);
  }

  /** Whether a selection still resolves in this graph. */
  static resolves(selection: Selection | null, graph: GraphData): boolean {
    if (!selection) return true;
    if (selection.kind === "entity") return graph.entities.some((e) => e.slug === selection.slug);
    if (selection.kind === "edge") return graph.edges.some((e) => e.key === selection.key);
    return graph.untyped.some((u) => u.id === selection.id);
  }

  private gone(text: string): void {
    this.container.createEl("p", { cls: "mp-muted", text });
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
      const live = edge.assertions.some((a) => a.status !== "dismissed");
      if (!live) continue;
      groups[edge.traversable ? "confirmed" : "proposed"].push(edge);
    }
    const untyped = graph.untyped.filter((u) => (u.source === slug || u.target === slug) && u.status !== "dismissed");
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

  private decisionButtons(parent: HTMLElement, assertion: string): void {
    const row = parent.createDiv({ cls: "mp-actions" });
    const reason = row.createEl("input", { cls: "mp-reason", attr: { placeholder: "reason (optional)" } });
    const confirm = row.createEl("button", { cls: "mp-confirm", text: "Confirm" });
    const dismiss = row.createEl("button", { cls: "mp-dismiss", text: "Dismiss" });
    const act = (action: Action) => async () => {
      confirm.disabled = dismiss.disabled = true;
      try {
        await this.actions.decide(assertion, action, reason.value.trim() || null);
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
