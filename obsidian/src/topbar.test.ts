import { describe, expect, it } from "vitest";

import { emptyGraph, type GraphAssertion, type GraphEdge, type GraphUntyped, type Status } from "./graph";
import { pendingCount } from "./topbar";

const assertion = (status: Status): GraphAssertion => ({
  id: `a_${status}_${Math.random()}`,
  note_id: "n",
  status,
  strength: 1,
  description: "",
  direction_corrected: false,
  text_unit_ids: [],
});
const edge = (key: string, ...statuses: Status[]): GraphEdge => ({
  key,
  source: "a",
  target: "b",
  type: "supports",
  proposed_type: null,
  directed: true,
  weight: 0,
  traversable: statuses.includes("confirmed"),
  assertions: statuses.map(assertion),
});
const untyped = (id: string, status: Status): GraphUntyped => ({
  id,
  source: "a",
  target: "b",
  proposed_type: "near",
  status,
  note_id: "n",
  description: "",
  text_unit_ids: [],
});

describe("pendingCount", () => {
  it("counts links with a proposal left to decide, typed and untyped", () => {
    const graph = {
      ...emptyGraph(),
      edges: [edge("e1", "proposed"), edge("e2", "confirmed", "proposed"), edge("e3", "confirmed"), edge("e4", "dismissed")],
      untyped: [untyped("u1", "proposed"), untyped("u2", "dismissed"), untyped("u3", "confirmed")],
    };
    expect(pendingCount(graph)).toBe(3);
  });
  it("is zero for an empty graph", () => {
    expect(pendingCount(emptyGraph())).toBe(0);
  });
});
