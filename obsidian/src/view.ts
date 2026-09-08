/** The Mind Palace view: wiring, polling, and the decision write. */

import { ItemView, Notice, Scope, type WorkspaceLeaf } from "obsidian";

import { GraphCanvas, type Selection, sameSelection } from "./canvas";
import { applyOverlay, decisionLine, foldDecisions, randomId, type Action } from "./decisions";
import { emptyGraph, parseGraph, type GraphData } from "./graph";
import { Panel, resolveSelection, type ReviewContext } from "./panel";
import { foldRetirements, pruneEntities, retireBlockers, retirementLine, type RetireAction } from "./retirements";
import { nextAfter, queueIndex, reviewQueue, step, type QueueItem } from "./review";
import { Topbar, type TopbarState } from "./topbar";

export const VIEW_TYPE = "mind-palace-graph";

export interface Settings {
  graphPath: string;
  decisionsPath: string;
  retirementsPath: string;
}

export const DEFAULT_SETTINGS: Settings = {
  graphPath: ".graph/graph.json",
  decisionsPath: ".mindpalace/decisions.jsonl",
  retirementsPath: ".mindpalace/retirements.jsonl",
};

const POLL_MS = 2000;
const UNDO_MS = 8000;

export class MindPalaceView extends ItemView {
  private canvas: GraphCanvas | null = null;
  private panel: Panel | null = null;
  private topbar: Topbar | null = null;
  private body: HTMLElement | null = null;
  private notice: HTMLElement | null = null;
  private graph: GraphData = emptyGraph();
  private selection: Selection | null = null;
  private filters: TopbarState | null = null;
  private stamps = { graph: -1, decisions: -1, retirements: -1 };
  // Reloads overlap: the poll and a decide's forced reload can be in flight
  // together, and a slower older read must not land on top of a newer one.
  private generation = 0;

  constructor(
    leaf: WorkspaceLeaf,
    private readonly settings: () => Settings,
  ) {
    super(leaf);
    this.bindKeys();
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
      // The hidden notice leaves auto-placement's second row vacant.
      // Keep the body in the flexible row even when the notice is absent.
      gridRow: "3",
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
      decide: (assertions, action, reason) => this.decide(assertions, action, reason),
      retire: (slug, reason) => this.retire(slug, reason),
    });
    await this.reload(true);
    this.registerInterval(window.setInterval(() => void this.reload(false), POLL_MS));
  }

  // Plain letters, active while the view is focused. A reason being typed
  // must keep its letters, so a focused field passes every key through.
  private bindKeys(): void {
    const scope = new Scope(this.app.scope);
    const bind = (key: string, run: () => void) =>
      scope.register([], key, () => {
        if (this.typing()) return;
        run();
        return false;
      });
    bind("c", () => void this.decideSelected("confirm"));
    bind("d", () => void this.decideSelected("dismiss"));
    bind("n", () => this.stepQueue(1));
    bind("p", () => this.stepQueue(-1));
    bind("r", () => void this.panel?.focusReason());
    bind("x", () => void this.retireSelected());
    this.scope = scope;
  }

  private typing(): boolean {
    const el = document.activeElement;
    return el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement;
  }

  async onClose(): Promise<void> {
    this.canvas?.destroy();
    this.canvas = null;
  }

  // ---- data ------------------------------------------------------------

  private async reload(force: boolean): Promise<void> {
    const generation = ++this.generation;
    const { graphPath, decisionsPath, retirementsPath } = this.settings();
    const adapter = this.app.vault.adapter;
    const [graphStat, decisionsStat, retirementsStat] = await Promise.all([
      adapter.stat(graphPath),
      adapter.stat(decisionsPath),
      adapter.stat(retirementsPath),
    ]);
    if (generation !== this.generation) return;
    if (!graphStat) {
      this.showNotice("No graph yet. Run the Mind Palace server once against this vault to write .graph/graph.json.");
      return;
    }
    const stamps = {
      graph: graphStat.mtime,
      decisions: decisionsStat?.mtime ?? 0,
      retirements: retirementsStat?.mtime ?? 0,
    };
    if (
      !force &&
      stamps.graph === this.stamps.graph &&
      stamps.decisions === this.stamps.decisions &&
      stamps.retirements === this.stamps.retirements
    ) {
      return;
    }
    this.stamps = stamps;
    let graph: GraphData;
    try {
      const [graphText, decisionsText, retirementsText] = await Promise.all([
        adapter.read(graphPath),
        decisionsStat ? adapter.read(decisionsPath) : Promise.resolve(""),
        retirementsStat ? adapter.read(retirementsPath) : Promise.resolve(""),
      ]);
      if (generation !== this.generation) return;
      // Overlay first, prune second: a dismissal made here must count when
      // deciding which implied entities remain (docs/decisions/0004).
      graph = pruneEntities(
        applyOverlay(parseGraph(graphText), foldDecisions(decisionsText)),
        foldRetirements(retirementsText),
      );
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
    this.panel?.show(this.selection, this.graph, this.filters?.asOf ?? null, this.review());
  }

  private reviewing(): boolean {
    return this.filters?.edgeMode === "proposed";
  }

  private queue(): QueueItem[] {
    return reviewQueue(this.graph, this.filters?.queueOrder ?? "graph");
  }

  private review(): ReviewContext {
    const queue = this.queue();
    return {
      queue,
      index: queueIndex(queue, this.selection),
      listing: this.reviewing(),
      grouped: this.filters?.queueOrder === "note",
    };
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

  private async appendDecision(assertions: string[], action: Action, reason: string | null): Promise<boolean> {
    const now = new Date();
    const lines = assertions.map((a) => decisionLine(a, action, reason, now, randomId())).join("");
    return this.appendLines(this.settings().decisionsPath, lines, "the decision");
  }

  private async appendLines(path: string, lines: string, what: string): Promise<boolean> {
    const adapter = this.app.vault.adapter;
    try {
      // One append is the smallest safe write there is, and a batch goes in
      // the same append so a crash cannot land half of it; the Python side
      // folds the file on its next open and re-derives everything.
      if (await adapter.exists(path)) await adapter.append(path, lines);
      else await adapter.write(path, lines);
      return true;
    } catch (error) {
      // Reported here; the panel's only job on failure is to hand the
      // buttons back, which it does whether or not this rejects.
      new Notice(`Mind Palace: could not write ${what}: ${(error as Error).message}`);
      return false;
    }
  }

  // ---- retiring (docs/decisions/0004 §Part 3) ----------------------------

  private async retire(slug: string, reason: string | null): Promise<void> {
    const blockers = retireBlockers(slug, this.graph);
    if (blockers) {
      new Notice(`Mind Palace: cannot retire ${slug}: ${blockers}.`);
      return;
    }
    if (!(await this.writeRetirement(slug, "retire", reason))) return;
    const fragment = document.createDocumentFragment();
    fragment.appendText(`Mind Palace: retired ${slug}`);
    const undo = fragment.createEl("button", { cls: "mp-undo", text: "Undo" });
    const notice = new Notice(fragment, UNDO_MS);
    undo.addEventListener("click", () => {
      notice.hide();
      void this.restore(slug);
    });
    // The node is gone from the canvas; the panel goes back to its lists.
    if (this.selection?.kind === "entity" && this.selection.slug === slug) this.onSelect(null);
    await this.reload(true);
  }

  private async restore(slug: string): Promise<void> {
    if (!(await this.writeRetirement(slug, "restore", null))) return;
    new Notice(`Mind Palace: restored ${slug}`);
    await this.reload(true);
    if (this.graph.entities.some((e) => e.slug === slug)) {
      this.onSelect({ kind: "entity", slug });
      this.canvas?.reveal({ kind: "entity", slug });
    }
  }

  private writeRetirement(slug: string, action: RetireAction, reason: string | null): Promise<boolean> {
    const line = retirementLine(slug, action, reason, new Date(), randomId());
    return this.appendLines(this.settings().retirementsPath, line, "the retirement");
  }

  private async retireSelected(): Promise<void> {
    if (this.selection?.kind !== "entity") return;
    const slug = this.selection.slug;
    await this.retire(slug, this.panel?.retireReasonFor(slug) ?? null);
  }

  private async decide(assertions: string[], action: Action, reason: string | null): Promise<void> {
    // Taken before the write: the decided item may leave the queue, and the
    // next item is defined relative to where it was.
    const before = this.queue();
    const index = queueIndex(before, this.selection);
    const item = before.find((q) => q.pending.includes(assertions[0]));
    if (!(await this.appendDecision(assertions, action, reason))) return;
    this.undoNotice(assertions, action, item?.label ?? assertions[0]);
    await this.reload(true);
    // Review mode walks on once the item has nothing left to decide. Any
    // other mode keeps the selection: jumping elsewhere would be a surprise.
    if (!this.reviewing() || index < 0) return;
    const after = this.queue();
    if (queueIndex(after, this.selection) >= 0) return;
    this.goTo(nextAfter(before, index, after));
  }

  private undoNotice(assertions: string[], action: Action, label: string): void {
    let verb = action === "confirm" ? "confirmed" : action === "dismiss" ? "dismissed" : "reopened";
    if (assertions.length > 1) verb += ` ${assertions.length} on`;
    // Built in full before the Notice takes it: the constructor moves the
    // fragment's children out, so anything added after would stand alone.
    const fragment = document.createDocumentFragment();
    fragment.appendText(`Mind Palace: ${verb} ${label}`);
    if (action === "reopen") {
      new Notice(fragment, UNDO_MS);
      return;
    }
    const undo = fragment.createEl("button", { cls: "mp-undo", text: "Undo" });
    const notice = new Notice(fragment, UNDO_MS);
    undo.addEventListener("click", () => {
      notice.hide();
      void this.undo(assertions);
    });
  }

  /** Reopen puts the assertions back under review and selects their item. */
  private async undo(assertions: string[]): Promise<void> {
    if (!(await this.appendDecision(assertions, "reopen", null))) return;
    await this.reload(true);
    const item = this.queue().find((q) => q.pending.includes(assertions[0]));
    this.undoNotice(assertions, "reopen", item?.label ?? assertions[0]);
    if (item) this.goTo(item);
  }

  private async decideSelected(action: Action): Promise<void> {
    const queue = this.queue();
    const item = queue[queueIndex(queue, this.selection)];
    if (!item) return;
    // The first pending card, with whatever reason was typed into it.
    const assertion = item.pending[0];
    await this.decide([assertion], action, this.panel?.reasonFor(assertion) ?? null);
  }

  private stepQueue(delta: 1 | -1): void {
    this.goTo(step(this.queue(), this.selection, delta));
  }

  private goTo(item: QueueItem | null): void {
    this.onSelect(item?.selection ?? null);
    if (item) this.canvas?.reveal(item.selection);
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
    this.canvas?.setFilters({ hiddenTypes: state.hiddenTypes, edgeMode: state.edgeMode, focus: state.focus });
    // A new focus selects its entity. Any other change (as-of, a type pill,
    // the edge mode) keeps whatever is selected and just re-renders.
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
