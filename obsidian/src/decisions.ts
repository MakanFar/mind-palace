/**
 * The decision overlay (docs/decisions/0003 §Part 2).
 *
 * `decisions.jsonl` is the one source of truth for review status, on both
 * sides. The plugin folds it the same way the Python `DecisionLog` does,
 * last write wins per assertion id, and writes lines in exactly the shape
 * `DecisionLog.append` writes. Nothing else is derived here.
 */

import type { GraphData } from "./graph";

export type Action = "confirm" | "dismiss" | "reopen";
export type Overlay = Map<string, Action>;

// "reopen" is the undo: the assertion goes back under review.
const STATUS = { confirm: "confirmed", dismiss: "dismissed", reopen: "proposed" } as const;

const REQUIRED_KEYS = ["op", "ts", "assertion", "action", "via"] as const;

export function foldDecisions(text: string): Overlay {
  const overlay: Overlay = new Map();
  // Python reads this file with `splitlines()`, under which a trailing
  // newline does not produce an empty last line. Match that, so "the final
  // line" means the same physical line on both sides.
  const lines = text.replace(/\n+$/, "").split("\n");
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let record: Record<string, unknown>;
    try {
      record = JSON.parse(line) as Record<string, unknown>;
    } catch {
      // A torn final line is a crash artefact from an interrupted append:
      // skip it. Anything earlier is real corruption and must be seen.
      if (index === lines.length - 1) return;
      throw new Error(`decisions.jsonl: malformed JSON at line ${index + 1}`);
    }
    // The same keys Python's DecisionLog.entries reads: a line missing one
    // would crash the server, so it must not quietly count here either.
    const missing = REQUIRED_KEYS.filter((key) => !(key in record));
    if (missing.length) {
      throw new Error(`decisions.jsonl: line ${index + 1} is missing ${missing.join(", ")}`);
    }
    if (typeof record.assertion === "string" && isAction(record.action)) {
      overlay.set(record.assertion, record.action);
    }
  });
  return overlay;
}

function isAction(value: unknown): value is Action {
  return value === "confirm" || value === "dismiss" || value === "reopen";
}

export function applyOverlay(graph: GraphData, overlay: Overlay): GraphData {
  const restatus = <T extends { id: string; status: string }>(item: T): T => {
    // "superseded" is derived by the Python fold from a *later* confirmed
    // claim, never from this log; the log still holds the old claim's own
    // confirm, which must not resurrect it.
    if (item.status === "superseded") return item;
    const action = overlay.get(item.id);
    return action ? { ...item, status: STATUS[action] } : item;
  };
  const edges = graph.edges.map((edge) => {
    const assertions = edge.assertions.map(restatus);
    // Same rule as the fold: traversable iff any member is confirmed, and
    // weight is the confirmed count. Derived from the statuses alone, so a
    // dismissal made here can retract what graph.json still says.
    const confirmed = assertions.filter((a) => a.status === "confirmed").length;
    return { ...edge, assertions, traversable: confirmed > 0, weight: confirmed };
  });
  // Same rule as the fold: a claim confirmed here retires the claim it
  // supersedes, so the panel strikes the old one through at once rather
  // than on the server's next sync.
  const claims = graph.claims.map(restatus);
  const ids = new Set(claims.map((c) => c.id));
  const retired = new Set(
    claims.filter((c) => c.status === "confirmed" && c.supersedes && ids.has(c.supersedes)).map((c) => c.supersedes),
  );
  return {
    ...graph,
    edges,
    untyped: graph.untyped.map(restatus),
    claims: claims.map((c) => (retired.has(c.id) ? { ...c, status: "superseded" as const } : c)),
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
  // Python reads the file with `splitlines()`, which also breaks on U+2028,
  // U+2029 and U+0085. JSON.stringify leaves those raw; escape them so a
  // reason typed in the panel can never split a record in two.
  const json = JSON.stringify(record).replace(/[\u2028\u2029\u0085]/g, (ch) =>
    "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0"),
  );
  return json + "\n";
}

export function randomId(): string {
  const alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
  let out = "";
  for (let i = 0; i < 16; i++) {
    out += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return out;
}
