/** The Mind Palace view: wiring, polling, and the decision write. */

import { ItemView, Notice, type WorkspaceLeaf } from "obsidian";

import { GraphCanvas, type Selection, sameSelection } from "./canvas";
import { applyOverlay, decisionLine, foldDecisions, randomId, type Action } from "./decisions";
import { emptyGraph, parseGraph, type GraphData } from "./graph";
import { Panel, resolveSelection } from "./panel";
import { Topbar, type TopbarState } from "./topbar";

export const VIEW_TYPE = "mind-palace-graph";

export interface Settings {
  graphPath: string;
  decisionsPath: string;
}

export const DEFAULT_SETTINGS: Settings = {
  graphPath: ".graph/graph.json",
  decisionsPath: ".mindpalace/decisions.jsonl",
};

const POLL_MS = 2000;

export class MindPalaceView extends ItemView {
  private canvas: GraphCanvas | null = null;
  private panel: Panel | null = null;
  private topbar: Topbar | null = null;
  private body: HTMLElement | null = null;
  private notice: HTMLElement | null = null;
  private graph: GraphData = emptyGraph();
  private selection: Selection | null = null;
  private filters: TopbarState | null = null;
  private stamps = { graph: -1, decisions: -1 };
  // Reloads overlap: the poll and a decide's forced reload can be in flight
  // together, and a slower older read must not land on top of a newer one.
  private generation = 0;

  constructor(
    leaf: WorkspaceLeaf,
    private readonly settings: () => Settings,
  ) {
    super(leaf);
  }

  getViewType(): string {
    return VIEW_TYPE;
  }

  getDisplayText(): string {
    return "Mind Palace";
  }

  getIcon(): string {
    return "network";
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("mp-root");
    // The sizing rules are set here as well as in styles.css. Obsidian
    // resolves the view content's percentage height as auto in some
    // layouts, and a stale stylesheet after a reload leaves the view
    // sized by the panel's content; inline styles hold either way.
    Object.assign(root.style, {
      position: "absolute",
      top: "var(--header-height)",
      left: "0",
      right: "0",
      bottom: "0",
      height: "auto",
      width: "auto",
      display: "grid",
      gridTemplateRows: "auto auto minmax(0, 1fr)",
      padding: "0",
      overflow: "hidden",
    } as Partial<CSSStyleDeclaration>);
    const top = root.createDiv();
    this.topbar = new Topbar(top, (state) => this.onFilters(state));
    this.notice = root.createDiv({ cls: "mp-notice" });
    this.notice.hide();
    this.body = root.createDiv({ cls: "mp-body" });
    Object.assign(this.body.style, {
      display: "grid",
      gridTemplateColumns: "minmax(0, 1fr) 320px",
      gridTemplateRows: "minmax(0, 1fr)",
      minHeight: "0",
    } as Partial<CSSStyleDeclaration>);
    const canvasHost = this.body.createDiv({ cls: "mp-canvas-host" });
    Object.assign(canvasHost.style, {
      position: "relative",
      minHeight: "0",
      minWidth: "0",
      overflow: "hidden",
    } as Partial<CSSStyleDeclaration>);
    const panelHost = this.body.createDiv();
    Object.assign(panelHost.style, { minHeight: "0", overflowY: "auto" } as Partial<CSSStyleDeclaration>);
    this.canvas = new GraphCanvas(canvasHost, {
      onSelect: (selection) => this.onSelect(selection),
    });
    this.panel = new Panel(panelHost, {
      openFile: (path) => void this.openFile(path),
      select: (selection) => this.onSelect(selection),
      decide: (assertion, action, reason) => this.decide(assertion, action, reason),
    });
    await this.reload(true);
    this.registerInterval(window.setInterval(() => void this.reload(false), POLL_MS));
  }

  async onClose(): Promise<void> {
    this.canvas?.destroy();
    this.canvas = null;
  }

  // ---- data ------------------------------------------------------------

  private async reload(force: boolean): Promise<void> {
    const generation = ++this.generation;
    const { graphPath, decisionsPath } = this.settings();
    const adapter = this.app.vault.adapter;
    const [graphStat, decisionsStat] = await Promise.all([adapter.stat(graphPath), adapter.stat(decisionsPath)]);
    if (generation !== this.generation) return;
    if (!graphStat) {
      this.showNotice("No graph yet. Run the Mind Palace server once against this vault to write .graph/graph.json.");
      return;
    }
    const stamps = { graph: graphStat.mtime, decisions: decisionsStat?.mtime ?? 0 };
    if (!force && stamps.graph === this.stamps.graph && stamps.decisions === this.stamps.decisions) return;
    this.stamps = stamps;
    let graph: GraphData;
    try {
      const [graphText, decisionsText] = await Promise.all([
        adapter.read(graphPath),
        decisionsStat ? adapter.read(decisionsPath) : Promise.resolve(""),
      ]);
      if (generation !== this.generation) return;
      graph = applyOverlay(parseGraph(graphText), foldDecisions(decisionsText));
    } catch (error) {
      if (generation !== this.generation) return;
      this.showNotice(`Could not read the graph: ${(error as Error).message}`);
      return;
    }
    this.graph = graph;
    this.hideNotice();
    this.topbar?.setGraph(this.graph);
    this.canvas?.setGraph(this.graph);
    if (this.filters) this.canvas?.setFilters(this.filters);
    if (this.selection && !resolveSelection(this.selection, this.graph)) {
      // A merge or adoption can retire the selected key between reloads.
      this.selection = null;
      this.canvas?.setSelection(null);
    }
    this.showPanel();
  }

  private showPanel(): void {
    this.panel?.show(this.selection, this.graph, this.filters?.asOf ?? null);
  }

  private async openFile(path: string): Promise<void> {
    // `openLinkText` creates a blank note for an unresolved path, which
    // would then parse as a malformed entity page. Entity pages only exist
    // after a rebuild, so check first.
    if (!(await this.app.vault.adapter.exists(path))) {
      new Notice(`Mind Palace: ${path} does not exist yet (run rebuild to write entity pages).`);
      return;
    }
    await this.app.workspace.openLinkText(path, "", true);
  }

  private async decide(assertion: string, action: Action, reason: string | null): Promise<void> {
    const { decisionsPath } = this.settings();
    const adapter = this.app.vault.adapter;
    const line = decisionLine(assertion, action, reason, new Date(), randomId());
    try {
      // One appended line is the smallest safe write there is; the Python
      // side folds the file on its next open and re-derives everything.
      if (await adapter.exists(decisionsPath)) await adapter.append(decisionsPath, line);
      else await adapter.write(decisionsPath, line);
    } catch (error) {
      // Reported here; the panel's only job on failure is to hand the
      // buttons back, which it does whether or not this rejects.
      new Notice(`Mind Palace: could not write the decision: ${(error as Error).message}`);
      return;
    }
    new Notice(`Mind Palace: ${action === "confirm" ? "confirmed" : "dismissed"} ${assertion}`);
    await this.reload(true);
  }

  // ---- interaction -----------------------------------------------------

  private onSelect(selection: Selection | null): void {
    if (sameSelection(selection, this.selection)) return;
    this.selection = selection;
    this.canvas?.setSelection(selection);
    this.showPanel();
  }

  private onFilters(state: TopbarState): void {
    const previousFocus = this.filters?.focus ?? null;
    this.filters = state;
    this.canvas?.setFilters({ hiddenTypes: state.hiddenTypes, showProposed: state.showProposed, focus: state.focus });
    // A new focus selects its entity. Any other change (as-of, a type pill,
    // the proposed toggle) keeps whatever is selected and just re-renders.
    if (state.focus && state.focus !== previousFocus) this.onSelect({ kind: "entity", slug: state.focus });
    else this.showPanel();
  }

  private showNotice(text: string): void {
    if (!this.notice) return;
    this.notice.setText(text);
    this.notice.show();
  }

  private hideNotice(): void {
    this.notice?.hide();
  }
}
