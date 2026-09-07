/** Search, type legend, proposed toggle, as-of (docs/decisions/0003 §Part 3). */

import type { GraphData } from "./graph";
import { colourFor } from "./palette";

export interface TopbarState {
  query: string;
  hiddenTypes: Set<string>;
  showProposed: boolean;
  asOf: string | null;
  focus: string | null;
}

let instances = 0;

export class Topbar {
  private readonly listId = `mp-entities-${++instances}`;
  private state: TopbarState = { query: "", hiddenTypes: new Set(), showProposed: true, asOf: null, focus: null };
  private readonly legend: HTMLElement;
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
      this.state.query = search.value;
      this.state.focus = this.slugs.has(search.value) ? search.value : null;
      this.emit();
    });
    search.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        search.value = "";
        this.state.query = "";
        this.state.focus = null;
        this.emit();
      }
    });
    this.legend = container.createDiv({ cls: "mp-legend" });
    const proposed = container.createEl("label", { cls: "mp-toggle" });
    const box = proposed.createEl("input", { attr: { type: "checkbox" } });
    box.checked = true;
    proposed.createSpan({ text: " proposed" });
    box.addEventListener("change", () => {
      this.state.showProposed = box.checked;
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

  get current(): TopbarState {
    return this.state;
  }

  setGraph(graph: GraphData): void {
    this.slugs = new Set(graph.entities.map((e) => e.slug));
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
