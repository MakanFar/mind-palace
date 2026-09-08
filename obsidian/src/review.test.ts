import { describe, expect, it } from "vitest";

import { emptyGraph, type GraphAssertion, type GraphData, type GraphEdge, type GraphUntyped, type Status } from "./graph";
import { nextAfter, queueIndex, reviewQueue, step } from "./review";

let n = 0;
const assertion = (status: Status, note = "n", strength = 1): GraphAssertion => ({
  id: `a_${++n}`,
  note_id: note,
  status,
  strength,
  description: "",
  direction_corrected: false,
  text_unit_ids: [],
});
const edge = (key: string, source: string, target: string, ...statuses: Status[]): GraphEdge => ({
  key,
  source,
  target,
  type: "supports",
  proposed_type: null,
  directed: true,
  weight: 0,
  traversable: statuses.includes("confirmed"),
  assertions: statuses.map((status) => assertion(status)),
});
const untyped = (id: string, status: Status): GraphUntyped => ({
  id,
  source: "c",
  target: "d",
  proposed_type: "near",
  status,
  note_id: "n",
  description: "",
  text_unit_ids: [],
});

const graph: GraphData = {
  ...emptyGraph(),
  edges: [
    edge("e1", "a", "b", "proposed"),
    edge("e2", "b", "c", "confirmed"),
    edge("e3", "a", "c", "confirmed", "proposed", "proposed"),
    edge("e4", "c", "d", "dismissed"),
  ],
  untyped: [untyped("u1", "proposed"), untyped("u2", "dismissed")],
};

describe("reviewQueue", () => {
  it("lists every link with a proposal left, in graph order, edges before untyped", () => {
    const queue = reviewQueue(graph);
    expect(queue.map((q) => q.selection)).toEqual([
      { kind: "edge", key: "e1" },
      { kind: "edge", key: "e3" },
      { kind: "untyped", id: "u1" },
    ]);
  });
  it("carries the pending assertion ids so a key press knows what to decide", () => {
    const [e1, e3, u1] = reviewQueue(graph);
    expect(e1.pending).toHaveLength(1);
    expect(e3.pending).toHaveLength(2);
    expect(u1.pending).toEqual(["u1"]);
  });
  it("labels items with both endpoints and the type", () => {
    const [e1, , u1] = reviewQueue(graph);
    expect(e1.label).toBe("a → b · supports");
    expect(u1.label).toBe("c — d · “near”");
  });
});

describe("queueIndex and step", () => {
  const queue = reviewQueue(graph);
  it("finds the selected item, or -1", () => {
    expect(queueIndex(queue, { kind: "edge", key: "e3" })).toBe(1);
    expect(queueIndex(queue, { kind: "edge", key: "e2" })).toBe(-1);
    expect(queueIndex(queue, null)).toBe(-1);
  });
  it("steps forward and back with wrap-around", () => {
    expect(step(queue, { kind: "edge", key: "e1" }, 1)?.selection).toEqual({ kind: "edge", key: "e3" });
    expect(step(queue, { kind: "untyped", id: "u1" }, 1)?.selection).toEqual({ kind: "edge", key: "e1" });
    expect(step(queue, { kind: "edge", key: "e1" }, -1)?.selection).toEqual({ kind: "untyped", id: "u1" });
  });
  it("starts from the first item when nothing in the queue is selected", () => {
    expect(step(queue, null, 1)?.selection).toEqual({ kind: "edge", key: "e1" });
    expect(step(queue, { kind: "edge", key: "e2" }, 1)?.selection).toEqual({ kind: "edge", key: "e1" });
    expect(step(queue, null, -1)?.selection).toEqual({ kind: "untyped", id: "u1" });
  });
  it("returns null for an empty queue", () => {
    expect(step([], null, 1)).toBeNull();
  });
});

describe("nextAfter", () => {
  const before = reviewQueue(graph);
  it("advances to the next surviving item after a decision removes the current one", () => {
    const after = reviewQueue({ ...graph, edges: graph.edges.filter((e) => e.key !== "e1") });
    expect(nextAfter(before, 0, after)?.selection).toEqual({ kind: "edge", key: "e3" });
  });
  it("skips items that were decided elsewhere in the meantime", () => {
    const after = reviewQueue({ ...graph, edges: graph.edges.filter((e) => e.key !== "e1" && e.key !== "e3") });
    expect(nextAfter(before, 0, after)?.selection).toEqual({ kind: "untyped", id: "u1" });
  });
  it("wraps to the start once the end is reached", () => {
    const after = reviewQueue({ ...graph, untyped: [] });
    expect(nextAfter(before, 2, after)?.selection).toEqual({ kind: "edge", key: "e1" });
  });
  it("is null once the queue is empty", () => {
    expect(nextAfter(before, 0, [])).toBeNull();
  });
});

describe("reviewQueue order", () => {
  const mk = (key: string, note: string, strength: number, ...more: GraphAssertion[]): GraphEdge => ({
    ...edge(key, "a", "b"),
    assertions: [assertion("proposed", note, strength), ...more],
  });
  const ordered: GraphData = {
    ...emptyGraph(),
    notes: { n_b: { path: "notes/beta.md" }, n_a: { path: "notes/alpha.md" } },
    edges: [
      mk("e1", "n_b", 3),
      mk("e2", "n_a", 9),
      mk("e3", "n_b", 7),
      // A confirmed member's strength does not count; only the pending ones do.
      mk("e4", "n_a", 1, assertion("confirmed", "n_a", 10)),
    ],
    untyped: [untyped("u1", "proposed")],
  };
  const keys = (order: "graph" | "note" | "strength") =>
    reviewQueue(ordered, order).map((q) => (q.selection.kind === "edge" ? q.selection.key : q.selection.kind === "untyped" ? q.selection.id : q.selection.slug));

  it("defaults to graph order", () => {
    expect(keys("graph")).toEqual(["e1", "e2", "e3", "e4", "u1"]);
    expect(reviewQueue(ordered).map((q) => q.selection)).toEqual(reviewQueue(ordered, "graph").map((q) => q.selection));
  });
  it("groups by note path, keeping graph order within a note", () => {
    // untyped u1 has note "n", which has no path, so it sorts by id after the paths.
    expect(keys("note")).toEqual(["e2", "e4", "e1", "e3", "u1"]);
  });
  it("carries the note path for the panel's group headings", () => {
    const [first] = reviewQueue(ordered, "note");
    expect(first.note).toBe("n_a");
    expect(first.notePath).toBe("notes/alpha.md");
  });
  it("sorts by the strongest pending assertion, strongest first, untyped last", () => {
    expect(keys("strength")).toEqual(["e2", "e3", "e1", "e4", "u1"]);
  });
});
