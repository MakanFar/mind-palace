/**
 * The decision overlay (docs/decisions/0003 §Part 2).
 *
 * `decisions.jsonl` is the one source of truth for review status, on both
 * sides. The plugin folds it the same way the Python `DecisionLog` does,
 * last write wins per assertion id, and writes lines in exactly the shape
 * `DecisionLog.append` writes. Nothing else is derived here.
 */

import type { GraphData } from "./graph";

export type Action = "confirm" | "dismiss";
export type Overlay = Map<string, Action>;

const STATUS = { confirm: "confirmed", dismiss: "dismissed" } as const;

export function foldDecisions(text: string): Overlay {
  const overlay: Overlay = new Map();
  const lines = text.split("\n");
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let record: { assertion?: unknown; action?: unknown };
    try {
      record = JSON.parse(line) as { assertion?: unknown; action?: unknown };
    } catch {
      // A torn final line is a crash artefact from an interrupted append:
      // skip it. Anything earlier is real corruption and must be seen.
      if (index === lines.length - 1) return;
      throw new Error(`decisions.jsonl: malformed JSON at line ${index + 1}`);
    }
    if (
      typeof record.assertion === "string" &&
      (record.action === "confirm" || record.action === "dismiss")
    ) {
      overlay.set(record.assertion, record.action);
    }
  });
  return overlay;
}

export function applyOverlay(graph: GraphData, overlay: Overlay): GraphData {
  const restatus = <T extends { id: string; status: string }>(item: T): T => {
    const action = overlay.get(item.id);
    return action ? { ...item, status: STATUS[action] } : item;
  };
  const edges = graph.edges.map((edge) => {
    const assertions = edge.assertions.map(restatus);
    return {
      ...edge,
      assertions,
      // The only structural fact the plugin derives: an edge becomes
      // traversable the moment any member assertion is confirmed.
      traversable: edge.traversable || assertions.some((a) => a.status === "confirmed"),
    };
  });
  return {
    ...graph,
    edges,
    untyped: graph.untyped.map(restatus),
    claims: graph.claims.map(restatus),
  };
}

export function decisionLine(
  assertion: string,
  action: Action,
  reason: string | null,
  now: Date,
  id: string,
): string {
  // Keys in sorted order, to match Python's `json.dumps(sort_keys=True)`.
  const record = {
    action,
    assertion,
    op: `obsidian_${id}`,
    reason,
    ts: now.toISOString().replace(/\.\d{3}Z$/, "Z"),
    via: "obsidian",
  };
  return JSON.stringify(record) + "\n";
}

export function randomId(): string {
  const alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
  let out = "";
  for (let i = 0; i < 16; i++) {
    out += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return out;
}
