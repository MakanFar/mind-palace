import { describe, expect, it } from "vitest";

import { emptyGraph, hasPending, parseGraph, type GraphAssertion, type GraphEdge, type Status } from "./graph";

describe("parseGraph", () => {
  it("accepts version 1 and rejects others", () => {
    const ok = parseGraph(JSON.stringify(emptyGraph()));
    expect(ok.entities).toEqual([]);
    expect(() => parseGraph(JSON.stringify({ version: 2 }))).toThrow(/version 2/);
  });
});

describe("hasPending", () => {
  const assertion = (status: Status): GraphAssertion => ({
    id: `a_${status}`,
    note_id: "n",
    status,
    strength: 1,
    description: "",
    direction_corrected: false,
    text_unit_ids: [],
  });
  const edge = (...statuses: Status[]): GraphEdge => ({
    key: "k",
    source: "a",
    target: "b",
    type: "supports",
    proposed_type: null,
    directed: true,
    weight: 0,
    traversable: statuses.includes("confirmed"),
    assertions: statuses.map(assertion),
  });
  it("is true while any member assertion is still proposed", () => {
    expect(hasPending(edge("proposed"))).toBe(true);
    expect(hasPending(edge("confirmed", "proposed"))).toBe(true);
  });
  it("is false once every member is decided", () => {
    expect(hasPending(edge("confirmed"))).toBe(false);
    expect(hasPending(edge("dismissed"))).toBe(false);
    expect(hasPending(edge("confirmed", "dismissed"))).toBe(false);
    expect(hasPending(edge())).toBe(false);
  });
});
