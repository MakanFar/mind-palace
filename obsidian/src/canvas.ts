/**
 * The graph canvas (docs/decisions/0003 §Part 3): d3-force on a 2D context.
 *
 * Draws confirmed edges solid, proposed dashed, untyped dotted, and nothing
 * for dismissed. Selection and focus dim everything else rather than hide
 * it, so a reader never loses the shape of the whole.
 */

import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";

import { type GraphData, type GraphEdge, isLive, isLiveEdge } from "./graph";
import { EDGE_STYLE, colourFor, radiusFor, type EdgeKind } from "./palette";

export type Selection =
  | { kind: "entity"; slug: string }
  | { kind: "edge"; key: string }
  | { kind: "untyped"; id: string };

export interface CanvasCallbacks {
  onSelect(selection: Selection | null): void;
}

export interface CanvasFilters {
  hiddenTypes: Set<string>;
  showProposed: boolean;
  focus: string | null;
}

interface Node extends SimulationNodeDatum {
  slug: string;
  type: string;
  rank: number;
  radius: number;
}

interface Link extends SimulationLinkDatum<Node> {
  id: string;
  kind: EdgeKind;
  directed: boolean;
  selection: Selection;
  sourceSlug: string;
  targetSlug: string;
}

const DIM = 0.08;
const UNSELECTED = 0.3;
const HIT_EDGE_PX = 6;

function cssVar(style: CSSStyleDeclaration, name: string, fallback: string): string {
  const value = style.getPropertyValue(name).trim();
  return value || fallback;
}

export class GraphCanvas {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private simulation: Simulation<Node, Link> | null = null;
  private nodes: Node[] = [];
  private links: Link[] = [];
  private byslug = new Map<string, Node>();
  private filters: CanvasFilters = { hiddenTypes: new Set(), showProposed: true, focus: null };
  private selection: Selection | null = null;
  private hovered: Selection | null = null;
  private transform = { x: 0, y: 0, k: 1 };
  private dragging: Node | null = null;
  private panning: { x: number; y: number; tx: number; ty: number } | null = null;
  private moved = false;
  private frame = 0;
  private readonly observer: ResizeObserver;
  private neighbours = new Map<string, Set<string>>();
  private signature = "";
  private emphasised: { nodes: Set<string> | null; links: Set<string> | null } | null = null;
  // True while the container has no size: the view was opened in a hidden
  // tab or a collapsed pane, and the layout ran against a 1x1 box.
  private degenerate = true;
  private readonly onMove = (event: MouseEvent): void => this.handleMove(event);
  private readonly onUp = (event: MouseEvent): void => this.handleUp(event);

  constructor(
    private readonly container: HTMLElement,
    private readonly callbacks: CanvasCallbacks,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "mp-canvas";
    container.appendChild(this.canvas);
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("mind-palace: canvas 2d context unavailable");
    this.ctx = ctx;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.bind();
  }

  // ---- data ------------------------------------------------------------

  setGraph(graph: GraphData): void {
    const previous = this.byslug;
    this.nodes = graph.entities.map((entity) => {
      const old = previous.get(entity.slug);
      return {
        slug: entity.slug,
        type: entity.type,
        rank: entity.rank,
        radius: radiusFor(entity.rank),
        x: old?.x,
        y: old?.y,
        vx: old?.vx ?? 0,
        vy: old?.vy ?? 0,
        fx: old?.fx,
        fy: old?.fy,
      };
    });
    this.byslug = new Map(this.nodes.map((n) => [n.slug, n]));
    // A reload during a drag must not leave the drag pinned to a node the
    // simulation no longer owns.
    if (this.dragging) this.dragging = this.byslug.get(this.dragging.slug) ?? null;
    this.links = [];
    for (const edge of graph.edges) {
      if (!this.byslug.has(edge.source) || !this.byslug.has(edge.target)) continue;
      if (!isLiveEdge(edge)) continue;
      this.links.push(this.link(edge));
    }
    for (const item of graph.untyped) {
      if (!isLive(item)) continue;
      if (!this.byslug.has(item.source) || !this.byslug.has(item.target)) continue;
      this.links.push({
        id: item.id,
        kind: "untyped",
        directed: false,
        selection: { kind: "untyped", id: item.id },
        sourceSlug: item.source,
        targetSlug: item.target,
        source: item.source,
        target: item.target,
      });
    }
    this.neighbours = new Map();
    this.emphasised = null;
    for (const link of this.links) {
      this.adjacent(link.sourceSlug).add(link.targetSlug);
      this.adjacent(link.targetSlug).add(link.sourceSlug);
    }
    // graph.json is rewritten on every server write, and the view polls it.
    // Only a change in *structure* is worth re-heating the layout; a status
    // flip or a new description just redraws in place.
    const signature = [
      this.nodes.map((n) => n.slug).sort().join(","),
      this.links.map((l) => `${l.id}:${l.sourceSlug}:${l.targetSlug}`).sort().join(","),
    ].join("|");
    if (signature !== this.signature) {
      this.signature = signature;
      this.restart(previous.size > 0 ? 0.4 : 1);
    } else {
      this.simulation?.nodes(this.nodes);
      (this.simulation?.force("link") as ReturnType<typeof forceLink<Node, Link>> | undefined)?.links(this.links);
      this.schedule();
    }
  }

  private link(edge: GraphEdge): Link {
    return {
      id: edge.key,
      kind: edge.traversable ? "confirmed" : "proposed",
      directed: edge.directed,
      selection: { kind: "edge", key: edge.key },
      sourceSlug: edge.source,
      targetSlug: edge.target,
      source: edge.source,
      target: edge.target,
    };
  }

  private adjacent(slug: string): Set<string> {
    let set = this.neighbours.get(slug);
    if (!set) {
      set = new Set();
      this.neighbours.set(slug, set);
    }
    return set;
  }

  setFilters(filters: CanvasFilters): void {
    const focusChanged = filters.focus !== this.filters.focus;
    this.filters = filters;
    this.emphasised = null;
    const node = focusChanged && filters.focus ? this.byslug.get(filters.focus) : undefined;
    if (node) {
      const { width, height } = this.size();
      const k = Math.max(this.transform.k, 1.4);
      this.transform = { k, x: width / 2 - (node.x ?? 0) * k, y: height / 2 - (node.y ?? 0) * k };
    }
    this.schedule();
  }

  setSelection(selection: Selection | null): void {
    this.selection = selection;
    this.emphasised = null;
    this.schedule();
  }

  destroy(): void {
    this.simulation?.stop();
    this.observer.disconnect();
    cancelAnimationFrame(this.frame);
    window.removeEventListener("mousemove", this.onMove);
    window.removeEventListener("mouseup", this.onUp);
    this.canvas.remove();
  }

  // ---- simulation ------------------------------------------------------

  private restart(alpha: number): void {
    this.simulation?.stop();
    const { width, height } = this.size();
    this.simulation = forceSimulation<Node, Link>(this.nodes)
      .alpha(alpha)
      .force(
        "link",
        forceLink<Node, Link>(this.links)
          .id((n) => n.slug)
          .distance(60),
      )
      .force("charge", forceManyBody<Node>().strength(-80))
      .force("center", forceCenter(width / 2, height / 2))
      .force("collide", forceCollide<Node>().radius((n) => n.radius + 2))
      .on("tick", () => this.schedule());
  }

  // ---- geometry --------------------------------------------------------

  private size(): { width: number; height: number } {
    const rect = this.container.getBoundingClientRect();
    return { width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
  }

  private resize(): void {
    const { width, height } = this.size();
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(width * ratio);
    this.canvas.height = Math.floor(height * ratio);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.simulation?.force("center", forceCenter(width / 2, height / 2));
    const wasDegenerate = this.degenerate;
    this.degenerate = width <= 1 || height <= 1;
    // Swapping the centre force moves nothing on its own: a cooled
    // simulation never ticks again. A layout that ran while hidden must be
    // re-heated when the pane first gets a real size, or it stays piled in
    // the top-left corner until someone drags a node.
    if (wasDegenerate && !this.degenerate) this.simulation?.alpha(1).restart();
    this.schedule();
  }

  private toGraph(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    return { x: (px - this.transform.x) / this.transform.k, y: (py - this.transform.y) / this.transform.k };
  }

  private hit(clientX: number, clientY: number): Selection | null {
    const p = this.toGraph(clientX, clientY);
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      if (!this.visible(n)) continue;
      const dx = (n.x ?? 0) - p.x;
      const dy = (n.y ?? 0) - p.y;
      if (dx * dx + dy * dy <= (n.radius + 2) * (n.radius + 2)) {
        return { kind: "entity", slug: n.slug };
      }
    }
    const tolerance = HIT_EDGE_PX / this.transform.k;
    for (const link of this.links) {
      if (!this.linkVisible(link)) continue;
      const a = link.source as Node;
      const b = link.target as Node;
      if (segmentDistance(p, a, b) <= tolerance) return link.selection;
    }
    return null;
  }

  private visible(node: Node): boolean {
    return !this.filters.hiddenTypes.has(node.type);
  }

  private linkVisible(link: Link): boolean {
    if (link.kind !== "confirmed" && !this.filters.showProposed) return false;
    return this.visible(link.source as Node) && this.visible(link.target as Node);
  }

  // ---- interaction -----------------------------------------------------

  private bind(): void {
    this.canvas.addEventListener("mousedown", (event) => {
      const hit = this.hit(event.clientX, event.clientY);
      this.moved = false;
      if (hit?.kind === "entity") {
        this.dragging = this.byslug.get(hit.slug) ?? null;
        if (this.dragging) {
          this.dragging.fx = this.dragging.x;
          this.dragging.fy = this.dragging.y;
          this.simulation?.alphaTarget(0.3).restart();
        }
      } else {
        this.panning = { x: event.clientX, y: event.clientY, tx: this.transform.x, ty: this.transform.y };
      }
    });
    window.addEventListener("mousemove", this.onMove);
    window.addEventListener("mouseup", this.onUp);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.0015);
      const next = Math.min(6, Math.max(0.2, this.transform.k * factor));
      const rect = this.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      this.transform.x = px - ((px - this.transform.x) * next) / this.transform.k;
      this.transform.y = py - ((py - this.transform.y) * next) / this.transform.k;
      this.transform.k = next;
      this.schedule();
    }, { passive: false });
  }

  private handleMove(event: MouseEvent): void {
    if (this.dragging) {
      const p = this.toGraph(event.clientX, event.clientY);
      this.dragging.fx = p.x;
      this.dragging.fy = p.y;
      this.moved = true;
      return;
    }
    if (this.panning) {
      this.transform.x = this.panning.tx + (event.clientX - this.panning.x);
      this.transform.y = this.panning.ty + (event.clientY - this.panning.y);
      this.moved = true;
      this.schedule();
      return;
    }
    const hit = this.hit(event.clientX, event.clientY);
    if (!sameSelection(hit, this.hovered)) {
      this.hovered = hit;
      this.canvas.style.cursor = hit ? "pointer" : "default";
      this.schedule();
    }
  }

  private handleUp(event: MouseEvent): void {
    if (this.dragging) {
      this.dragging.fx = null;
      this.dragging.fy = null;
      this.simulation?.alphaTarget(0);
      if (!this.moved) this.callbacks.onSelect({ kind: "entity", slug: this.dragging.slug });
      this.dragging = null;
      return;
    }
    if (this.panning) {
      this.panning = null;
      // The view owns the selection and pushes it back via setSelection.
      if (!this.moved) this.callbacks.onSelect(this.hit(event.clientX, event.clientY));
    }
  }

  // ---- drawing ---------------------------------------------------------

  private schedule(): void {
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => this.draw());
  }

  private emphasis(): { nodes: Set<string> | null; links: Set<string> | null } {
    // Which slugs and link ids are "in": everything else is dimmed. Depends
    // only on selection, focus and links, so it is kept until one changes
    // rather than rebuilt on every frame of the simulation.
    if (this.emphasised) return this.emphasised;
    const focusSet = this.filters.focus ? this.twoHop(this.filters.focus) : null;
    let selected: Set<string> | null = null;
    let selectedLinks: Set<string> | null = null;
    if (this.selection?.kind === "entity") {
      const slug = this.selection.slug;
      selected = new Set([slug, ...this.adjacent(slug)]);
      selectedLinks = new Set(
        this.links.filter((l) => l.sourceSlug === slug || l.targetSlug === slug).map((l) => l.id),
      );
    } else if (this.selection) {
      const id = this.selection.kind === "edge" ? this.selection.key : this.selection.id;
      const link = this.links.find((l) => l.id === id);
      if (link) {
        selected = new Set([link.sourceSlug, link.targetSlug]);
        selectedLinks = new Set([link.id]);
      }
    }
    const nodes = focusSet && selected ? new Set([...focusSet].filter((s) => selected!.has(s))) : focusSet ?? selected;
    const links = selectedLinks ?? (focusSet ? new Set(this.links.filter((l) => focusSet.has(l.sourceSlug) && focusSet.has(l.targetSlug)).map((l) => l.id)) : null);
    this.emphasised = { nodes, links };
    return this.emphasised;
  }

  private twoHop(slug: string): Set<string> {
    const out = new Set([slug]);
    for (const first of this.adjacent(slug)) {
      out.add(first);
      for (const second of this.adjacent(first)) out.add(second);
    }
    return out;
  }

  private draw(): void {
    const { width, height } = this.size();
    const ctx = this.ctx;
    // One style resolution per frame, not one per variable.
    const style = getComputedStyle(this.container);
    const accent = cssVar(style, "--interactive-accent", "#7c6cf0");
    const text = cssVar(style, "--text-normal", "#e0e0e0");
    const muted = cssVar(style, "--text-muted", "#999");
    const font = cssVar(style, "--font-interface", "sans-serif");
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(this.transform.x, this.transform.y);
    ctx.scale(this.transform.k, this.transform.k);

    const { nodes: inNodes, links: inLinks } = this.emphasis();
    const hoveredId =
      this.hovered?.kind === "edge" ? this.hovered.key : this.hovered?.kind === "untyped" ? this.hovered.id : null;
    const hoveredSlug = this.hovered?.kind === "entity" ? this.hovered.slug : null;

    for (const link of this.links) {
      if (!this.linkVisible(link)) continue;
      const a = link.source as Node;
      const b = link.target as Node;
      const style = EDGE_STYLE[link.kind];
      const emphasised = inLinks ? inLinks.has(link.id) : true;
      const dim = this.filters.focus && !emphasised ? DIM : !emphasised ? UNSELECTED : 1;
      ctx.globalAlpha = style.alpha * dim;
      ctx.strokeStyle = inLinks?.has(link.id) && this.selection ? accent : muted;
      ctx.lineWidth = (link.id === hoveredId ? 2.5 : 1.2) / this.transform.k;
      ctx.setLineDash(style.dash.map((d) => d / this.transform.k));
      ctx.beginPath();
      ctx.moveTo(a.x ?? 0, a.y ?? 0);
      ctx.lineTo(b.x ?? 0, b.y ?? 0);
      ctx.stroke();
      if (link.directed) this.arrow(a, b, ctx);
    }
    ctx.setLineDash([]);

    for (const node of this.nodes) {
      if (!this.visible(node)) continue;
      const emphasised = inNodes ? inNodes.has(node.slug) : true;
      ctx.globalAlpha = this.filters.focus && !emphasised ? DIM : !emphasised ? UNSELECTED : 1;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = colourFor(node.type);
      ctx.fill();
      const isSelected = this.selection?.kind === "entity" && this.selection.slug === node.slug;
      if (isSelected || node.slug === hoveredSlug) {
        ctx.lineWidth = 2 / this.transform.k;
        ctx.strokeStyle = accent;
        ctx.stroke();
      }
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = text;
    // A `var()` inside ctx.font is silently rejected and the default 10px
    // font kept; resolve the family the same way the colours are resolved.
    ctx.font = `${11 / this.transform.k}px ${font}`;
    for (const node of this.nodes) {
      if (!this.visible(node)) continue;
      const emphasised = inNodes ? inNodes.has(node.slug) : false;
      const show =
        node.slug === hoveredSlug || emphasised || (this.transform.k > 1.2 && node.rank >= 3);
      if (!show) continue;
      ctx.globalAlpha = this.filters.focus && !emphasised && node.slug !== hoveredSlug ? DIM : 1;
      ctx.fillText(node.slug, (node.x ?? 0) + node.radius + 3 / this.transform.k, (node.y ?? 0) + 4 / this.transform.k);
    }
    ctx.restore();
  }

  private arrow(a: Node, b: Node, ctx: CanvasRenderingContext2D): void {
    const dx = (b.x ?? 0) - (a.x ?? 0);
    const dy = (b.y ?? 0) - (a.y ?? 0);
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const tipX = (b.x ?? 0) - ux * (b.radius + 2);
    const tipY = (b.y ?? 0) - uy * (b.radius + 2);
    const size = 5 / this.transform.k;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ux * size - uy * size * 0.6, tipY - uy * size + ux * size * 0.6);
    ctx.lineTo(tipX - ux * size + uy * size * 0.6, tipY - uy * size - ux * size * 0.6);
    ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }
}

function segmentDistance(p: { x: number; y: number }, a: Node, b: Node): number {
  const ax = a.x ?? 0;
  const ay = a.y ?? 0;
  const bx = b.x ?? 0;
  const by = b.y ?? 0;
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p.x - ax) * dx + (p.y - ay) * dy) / len2));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(p.x - cx, p.y - cy);
}

export function sameSelection(a: Selection | null, b: Selection | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === "entity" && b.kind === "entity") return a.slug === b.slug;
  if (a.kind === "edge" && b.kind === "edge") return a.key === b.key;
  if (a.kind === "untyped" && b.kind === "untyped") return a.id === b.id;
  return false;
}
