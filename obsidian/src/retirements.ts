/**
 * The retirement overlay (docs/decisions/0004 §Part 3).
 *
 * `retirements.jsonl` is folded the way the Python `RetirementLog` folds it,
 * last write wins per slug, and lines are written in exactly the shape
 * `RetirementLog.append` writes. The prune rule is the fold's own, so a
 * retirement or a dismissal made here takes the node off the canvas before
 * the Python side re-folds.
 */

import { type GraphData, isLive, liveSlugs } from "./graph";

export type RetireAction = "retire" | "restore";

const REQUIRED_KEYS = ["op", "ts", "slug", "action", "via"] as const;

export function foldRetirements(text: string): Set<string> {
  const retired = new Set<string>();
  // Same line rules as decisions.jsonl: a torn final line is a crash
  // artefact and is skipped; anything earlier is corruption and is raised.
  const lines = text.replace(/\n+$/, "").split("\n");
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let record: Record<string, unknown>;
    try {
      record = JSON.parse(line) as Record<string, unknown>;
    } catch {
      if (index === lines.length - 1) return;
      throw new Error(`retirements.jsonl: malformed JSON at line ${index + 1}`);
    }
    const missing = REQUIRED_KEYS.filter((key) => !(key in record));
    if (missing.length) {
      throw new Error(`retirements.jsonl: line ${index + 1} is missing ${missing.join(", ")}`);
    }
    if (typeof record.slug !== "string") return;
    if (record.action === "retire") retired.add(record.slug);
    else if (record.action === "restore") retired.delete(record.slug);
  });
  return retired;
}

/** The fold's rule: an entity stays when it is declared or live, and a
 *  retired one only when live. Run after `applyOverlay`, so a dismissal made
 *  in the window counts. */
export function pruneEntities(graph: GraphData, retired: Set<string>): GraphData {
  const live = liveSlugs(graph);
  return {
    ...graph,
    entities: graph.entities.filter((e) => {
      if (live.has(e.slug)) return true;
      return (e.declared ?? true) && !retired.has(e.slug);
    }),
  };
}

/** Why retire would be refused, in the tool's words, or null. */
export function retireBlockers(slug: string, graph: GraphData): string | null {
  const counts = new Map<string, number>();
  const add = (status: string, kind: string) => {
    const key = `${status} ${kind}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  };
  for (const edge of graph.edges) {
    if (edge.source !== slug && edge.target !== slug) continue;
    for (const a of edge.assertions) if (isLive(a)) add(a.status, "relationship");
  }
  for (const item of graph.untyped) {
    if ((item.source === slug || item.target === slug) && isLive(item)) add(item.status, "relationship");
  }
  for (const claim of graph.claims) {
    if (claim.subject === slug && isLive(claim)) add(claim.status, "claim");
  }
  if (!counts.size) return null;
  const parts: string[] = [];
  for (const [key, count] of counts) {
    const [status, kind] = key.split(" ");
    const verb = status === "confirmed" ? "dismiss" : "decide";
    parts.push(`${count} ${status} ${kind}${count === 1 ? "" : "s"}: ${verb} ${count === 1 ? "it" : "them"} first`);
  }
  return parts.join("; ");
}

export function retirementLine(
  slug: string,
  action: RetireAction,
  reason: string | null,
  now: Date,
  id: string,
): string {
  // Keys in sorted order, to match Python's `json.dumps(sort_keys=True)`.
  const record = {
    action,
    op: `obsidian_${id}`,
    reason,
    slug,
    ts: now.toISOString().replace(/\.\d{3}Z$/, "Z"),
    via: "obsidian",
  };
  const json = JSON.stringify(record).replace(/[\u2028\u2029\u0085]/g, (ch) =>
    "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0"),
  );
  return json + "\n";
}
