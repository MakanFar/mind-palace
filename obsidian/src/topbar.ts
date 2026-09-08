/** Search, type legend, edge mode, as-of (docs/decisions/0003 §Part 3). */

import type { EdgeMode } from "./canvas";
import type { GraphData } from "./graph";
import { colourFor } from "./palette";
import { type QueueOrder, reviewQueue } from "./review";

export interface TopbarState {
  hiddenTypes: Set<string>;
  edgeMode: EdgeMode;
  queueOrder: QueueOrder;
  asOf: string | null;
  focus: string | null;
}

const QUEUE_ORDERS: Record<QueueOrder, string> = {
  graph: "graph order",
  note: "by note",
  strength: "by strength",
};

const EDGE_MODES: Record<EdgeMode, string> = {
  all: "all edges",
  confirmed: "confirmed only",
  proposed: "proposed only",
};

/** How many links review mode would draw: the size of the review queue. */
export function pendingCount(graph: GraphData): number {
  return reviewQueue(graph).length;
}

let instances = 0;

export class Topbar {
  private readonly listId = `mp-entities-${++instances}`;
  private state: TopbarState = { hiddenTypes: new Set(), edgeMode: "all", queueOrder: "graph", asOf: null, focus: null };
  private readonly legend: HTMLElement;
  private readonly proposedOption: HTMLOptionElement;
  private readonly datalist: HTMLDataListElement;
  private slugs = new Set<string>();

  constructor(
    container: HTMLElement,
    private readonly onChange: (state: TopbarState) => void,
  ) {
    container.classList.add("mp-topbar");
    const search = container.createEl("input", {
      cls: "mp-search",
      attr: { type: "search", placeholder: "focus an entity…", list: this.listId },
    });
    this.datalist = container.createEl("datalist", { attr: { id: this.listId } });
    search.addEventListener("input", () => {
      this.state.focus = this.slugs.has(search.value) ? search.value : null;
      this.emit();
    });
    search.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        search.value = "";
        this.state.focus = null;
        this.emit();
      }
    });
    this.legend = container.createDiv({ cls: "mp-legend" });
    const edges = container.createEl("label", { cls: "mp-toggle" });
    edges.createSpan({ text: "edges " });
    const mode = edges.createEl("select", { cls: "dropdown" });
    const options = new Map<EdgeMode, HTMLOptionElement>();
    for (const [value, text] of Object.entries(EDGE_MODES) as [EdgeMode, string][]) {
      options.set(value, mode.createEl("option", { text, attr: { value } }));
    }
    this.proposedOption = options.get("proposed")!;
    // The queue order only means something while the queue is on screen.
    const order = container.createEl("label", { cls: "mp-toggle" });
    order.createSpan({ text: "order " });
    const orderSelect = order.createEl("select", { cls: "dropdown" });
    for (const [value, text] of Object.entries(QUEUE_ORDERS) as [QueueOrder, string][]) {
      orderSelect.createEl("option", { text, attr: { value } });
    }
    order.hide();
    orderSelect.addEventListener("change", () => {
      this.state.queueOrder = orderSelect.value as QueueOrder;
      this.emit();
    });
    mode.addEventListener("change", () => {
      this.state.edgeMode = mode.value as EdgeMode;
      order.toggle(this.state.edgeMode === "proposed");
      this.emit();
    });
    const asOf = container.createEl("label", { cls: "mp-toggle" });
    asOf.createSpan({ text: "as of " });
    const date = asOf.createEl("input", { attr: { type: "date" } });
    date.addEventListener("change", () => {
      this.state.asOf = date.value || null;
      this.emit();
    });
  }

  setGraph(graph: GraphData): void {
    this.slugs = new Set(graph.entities.map((e) => e.slug));
    // The queue length sits on the option so it is readable before and
    // after switching into review mode.
    this.proposedOption.text = `${EDGE_MODES.proposed} (${pendingCount(graph)})`;
    // A focus whose entity a merge or rename retired would otherwise dim the
    // whole canvas around a node that no longer exists.
    if (this.state.focus && !this.slugs.has(this.state.focus)) {
      this.state.focus = null;
      this.emit();
    }
    this.datalist.empty();
    for (const slug of [...this.slugs].sort()) this.datalist.createEl("option", { attr: { value: slug } });
    const counts = new Map<string, number>();
    for (const entity of graph.entities) counts.set(entity.type, (counts.get(entity.type) ?? 0) + 1);
    // By count, not by first appearance: the few visible pills should be the
    // types that dominate the picture, and ties sort by name so the order
    // does not jitter between reloads.
    const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    this.legend.empty();
    for (const [type, count] of ordered) {
      const pill = this.legend.createEl("button", { cls: "mp-pill" });
      const dot = pill.createSpan({ cls: "mp-dot" });
      dot.style.background = colourFor(type);
      pill.createSpan({ text: ` ${type} ${count}` });
      if (this.state.hiddenTypes.has(type)) pill.addClass("mp-pill-off");
      pill.addEventListener("click", () => {
        if (this.state.hiddenTypes.has(type)) this.state.hiddenTypes.delete(type);
        else this.state.hiddenTypes.add(type);
        pill.toggleClass("mp-pill-off", this.state.hiddenTypes.has(type));
        this.emit();
      });
    }
  }

  private emit(): void {
    this.onChange({ ...this.state, hiddenTypes: new Set(this.state.hiddenTypes) });
  }
}
