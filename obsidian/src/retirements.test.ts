import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { emptyGraph, isolatedEntities, liveSlugs, type GraphAssertion, type GraphData, type GraphEdge, type GraphEntity, type Status } from "./graph";
import { foldRetirements, pruneEntities, retireBlockers, retirementLine } from "./retirements";

const entity = (slug: string, declared: boolean): GraphEntity => ({
  slug,
  type: "concept",
  rank: 0,
  merged_from: [],
  declared,
  description: "",
  stale: true,
  note_ids: [],
  text_unit_ids: [],
});
let n = 0;
const assertion = (status: Status): GraphAssertion => ({
  id: `a_${++n}`,
  note_id: "n",
  status,
  strength: 1,
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
  assertions: statuses.map((s) => assertion(s)),
});

// a: declared, isolated. b: declared, confirmed edge to c. c: implied by that
// edge. d: implied only by a dismissed edge to e. e: declared. f: implied by a
// proposed claim's subject. g: implied by a proposed untyped assertion.
const graph: GraphData = {
  ...emptyGraph(),
  entities: ["a", "e"].map((s) => entity(s, true)).concat([entity("b", true), entity("c", false), entity("d", false), entity("f", false), entity("g", false)]),
  edges: [edge("e1", "b", "c", "confirmed"), edge("e2", "d", "e", "dismissed")],
  claims: [{ id: "k_1", subject: "f", text: "t", status: "proposed", valid_from: null, valid_to: null, supersedes: null, note_id: "n", text_unit_ids: [] }],
  untyped: [{ id: "u_1", source: "g", target: "e", proposed_type: "near", status: "proposed", note_id: "n", description: "", text_unit_ids: [] }],
};

describe("liveSlugs and isolatedEntities", () => {
  it("live means named by something not dismissed", () => {
    expect([...liveSlugs(graph)].sort()).toEqual(["b", "c", "e", "f", "g"]);
  });
  it("isolated entities have nothing live attached", () => {
    expect(isolatedEntities(graph).map((e) => e.slug)).toEqual(["a", "d"]);
  });
});

describe("pruneEntities", () => {
  it("drops an undeclared entity named only by dismissed items", () => {
    const out = pruneEntities(graph, new Set());
    expect(out.entities.map((e) => e.slug).sort()).toEqual(["a", "b", "c", "e", "f", "g"]);
  });
  it("drops a retired declared entity but keeps a retired one something live names", () => {
    const out = pruneEntities(graph, new Set(["a", "b"]));
    expect(out.entities.map((e) => e.slug).sort()).toEqual(["b", "c", "e", "f", "g"]);
  });
  it("treats a missing declared field as declared, for an older graph.json", () => {
    const old = { ...graph, entities: graph.entities.map(({ declared: _d, ...rest }) => rest as GraphEntity) };
    expect(pruneEntities(old, new Set()).entities.map((e) => e.slug)).toContain("d");
  });
  it("leaves the input untouched", () => {
    pruneEntities(graph, new Set(["a"]));
    expect(graph.entities.map((e) => e.slug)).toContain("a");
  });
});

describe("retireBlockers", () => {
  it("is null for an isolated entity", () => {
    expect(retireBlockers("a", graph)).toBeNull();
  });
  it("names what stands in the way, in the tool's words", () => {
    expect(retireBlockers("b", graph)).toBe("1 confirmed relationship: dismiss it first");
    expect(retireBlockers("f", graph)).toBe("1 proposed claim: decide it first");
    expect(retireBlockers("g", graph)).toBe("1 proposed relationship: decide it first");
  });
  it("lists several kinds", () => {
    const g = { ...graph, edges: [...graph.edges, edge("e3", "f", "a", "proposed", "proposed")] };
    expect(retireBlockers("f", g)).toBe("2 proposed relationships: decide them first; 1 proposed claim: decide it first");
  });
});

describe("foldRetirements", () => {
  const fixture = readFileSync(resolve(__dirname, "../../tests/fixtures/retirements-conformance.jsonl"), "utf8");
  it("folds the shared fixture to the same set as Python, torn final line included", () => {
    expect([...foldRetirements(fixture)]).toEqual(["b"]);
  });
  it("throws on a malformed interior line", () => {
    expect(() => foldRetirements('{"bad"\n{"action":"retire","slug":"a","op":"o","ts":"t","via":"v"}\n')).toThrow(/line 1/);
  });
  it("is empty for an empty file", () => {
    expect(foldRetirements("").size).toBe(0);
  });
});

describe("retirementLine", () => {
  it("matches the Python RetirementLog shape with sorted keys", () => {
    const line = retirementLine("typo", "retire", "not a thing", new Date(Date.UTC(2026, 8, 7, 10, 0, 0)), "abc");
    expect(line).toBe(
      '{"action":"retire","op":"obsidian_abc","reason":"not a thing","slug":"typo","ts":"2026-09-07T10:00:00Z","via":"obsidian"}\n',
    );
  });
  it("round-trips through foldRetirements", () => {
    const text = retirementLine("x", "retire", null, new Date(), "1") + retirementLine("x", "restore", null, new Date(), "2");
    expect(foldRetirements(text).size).toBe(0);
  });
});
