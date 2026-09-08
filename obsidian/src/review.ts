/**
 * The review queue: every link with a proposal still to decide, in graph
 * order. The panel lists it, the keyboard walks it, and a decision moves
 * on to the next surviving item.
 */

import { type Selection, sameSelection } from "./canvas";
import { type GraphData, hasPending, isLive, isLiveEdge } from "./graph";

export interface QueueItem {
  selection: Selection;
  label: string;
  /** Assertion ids still proposed, in the order the panel shows them. */
  pending: string[];
  /** The note the first pending assertion came from, for grouping. */
  note: string;
  notePath: string | null;
  /** The strongest pending assertion; untyped ones carry no strength. */
  strength: number;
}

/**
 * "graph" is the export's order. "note" groups a note's extractions so they
 * are reviewed with that note in mind. "strength" puts the confident ones
 * first for quick confirms and gathers the doubtful tail for dismissals.
 */
export type QueueOrder = "graph" | "note" | "strength";

export function reviewQueue(graph: GraphData, order: QueueOrder = "graph"): QueueItem[] {
  const queue: QueueItem[] = [];
  const notePath = (note: string) => graph.notes[note]?.path ?? null;
  for (const edge of graph.edges) {
    if (!isLiveEdge(edge) || !hasPending(edge)) continue;
    const pending = edge.assertions.filter((a) => a.status === "proposed");
    queue.push({
      selection: { kind: "edge", key: edge.key },
      label: `${edge.source} ${edge.directed ? "→" : "—"} ${edge.target} · ${edge.type}`,
      pending: pending.map((a) => a.id),
      note: pending[0].note_id,
      notePath: notePath(pending[0].note_id),
      strength: Math.max(...pending.map((a) => a.strength)),
    });
  }
  for (const item of graph.untyped) {
    if (!isLive(item) || item.status !== "proposed") continue;
    queue.push({
      selection: { kind: "untyped", id: item.id },
      label: `${item.source} — ${item.target} · “${item.proposed_type ?? "?"}”`,
      pending: [item.id],
      note: item.note_id,
      notePath: notePath(item.note_id),
      strength: 0,
    });
  }
  // Array.prototype.sort is stable: ties keep graph order.
  if (order === "note") queue.sort((a, b) => noteKey(a).localeCompare(noteKey(b)));
  else if (order === "strength") queue.sort((a, b) => b.strength - a.strength);
  return queue;
}

function noteKey(item: QueueItem): string {
  // Paths sort before bare ids so notes the export knows come first.
  return item.notePath ? `0${item.notePath}` : `1${item.note}`;
}

export function queueIndex(queue: QueueItem[], selection: Selection | null): number {
  return queue.findIndex((item) => sameSelection(item.selection, selection));
}

/** The item `delta` places along from the selection, wrapping; from the
 *  first (or last) when nothing in the queue is selected. */
export function step(queue: QueueItem[], selection: Selection | null, delta: 1 | -1): QueueItem | null {
  if (!queue.length) return null;
  const index = queueIndex(queue, selection);
  if (index < 0) return delta > 0 ? queue[0] : queue[queue.length - 1];
  return queue[(index + delta + queue.length) % queue.length];
}

/** After the item at `index` of `previous` was decided: the next item after
 *  it that is still in `current`, else the first of `current`. */
export function nextAfter(previous: QueueItem[], index: number, current: QueueItem[]): QueueItem | null {
  for (let i = index + 1; i < previous.length; i++) {
    const survivor = current.find((item) => sameSelection(item.selection, previous[i].selection));
    if (survivor) return survivor;
  }
  return current[0] ?? null;
}
