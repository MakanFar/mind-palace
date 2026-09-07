/** The Mind Palace view: wiring, polling, and the decision write. */

import { ItemView, Notice, type WorkspaceLeaf } from "obsidian";

import { GraphCanvas, type Selection, sameSelection } from "./canvas";
import { applyOverlay, decisionLine, foldDecisions, randomId, type Action } from "./decisions";
import { emptyGraph, parseGraph, type GraphData } from "./graph";
import { Panel } from "./panel";
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
  private panelSnapshot = "";

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
    const top = root.createDiv();
    this.topbar = new Topbar(top, (state) => this.onFilters(state));
    this.notice = root.createDiv({ cls: "mp-notice" });
    this.notice.hide();
    this.body = root.createDiv({ cls: "mp-body" });
    const canvasHost = this.body.createDiv({ cls: "mp-canvas-host" });
    const panelHost = this.body.createDiv();
    this.canvas = new GraphCanvas(canvasHost, {
      onSelect: (selection) => this.onSelect(selection),
      onHover: () => undefined,
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
    const { graphPath, decisionsPath } = this.settings();
    const adapter = this.app.vault.adapter;
    const graphStat = await adapter.stat(graphPath);
    if (!graphStat) {
      this.showNotice("No graph yet. Run the Mind Palace server once against this vault to write .graph/graph.json.");
      return;
    }
    const decisionsStat = await adapter.stat(decisionsPath);
    const stamps = { graph: graphStat.mtime, decisions: decisionsStat?.mtime ?? 0 };
    if (!force && stamps.graph === this.stamps.graph && stamps.decisions === this.stamps.decisions) return;
    this.stamps = stamps;
    try {
      const base = parseGraph(await adapter.read(graphPath));
      const overlay = decisionsStat ? foldDecisions(await adapter.read(decisionsPath)) : new Map();
      this.graph = applyOverlay(base, overlay);
      this.hideNotice();
    } catch (error) {
      this.showNotice(`Could not read the graph: ${(error as Error).message}`);
      return;
    }
    this.topbar?.setGraph(this.graph);
    this.canvas?.setGraph(this.graph);
    if (this.filters) this.canvas?.setFilters(this.filters);
    if (!Panel.resolves(this.selection, this.graph)) {
      // A merge or adoption can retire the selected key between reloads.
      this.selection = null;
      this.canvas?.setSelection(null);
    }
    this.showPanel();
  }

  private showPanel(): void {
    const asOf = this.filters?.asOf ?? null;
    const snapshot = Panel.snapshot(this.selection, this.graph, asOf);
    if (snapshot === this.panelSnapshot) return;
    this.panelSnapshot = snapshot;
    this.panel?.show(this.selection, this.graph, asOf);
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
      new Notice(`Mind Palace: could not write the decision: ${(error as Error).message}`);
      throw error;
    }
    new Notice(`Mind Palace: ${action === "confirm" ? "confirmed" : "dismissed"} ${assertion}`);
    await this.reload(true);
  }

  // ---- interaction -----------------------------------------------------

  private onSelect(selection: Selection | null): void {
    if (sameSelection(selection, this.selection) && selection !== null) return;
    this.selection = selection;
    this.canvas?.setSelection(selection);
    this.showPanel();
  }

  private onFilters(state: TopbarState): void {
    this.filters = state;
    this.canvas?.setFilters({ hiddenTypes: state.hiddenTypes, showProposed: state.showProposed, focus: state.focus });
    if (state.focus) this.onSelect({ kind: "entity", slug: state.focus });
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
