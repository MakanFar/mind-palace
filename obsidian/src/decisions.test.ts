import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { applyOverlay, decisionLine, foldDecisions } from "./decisions";
import { emptyGraph, type GraphData } from "./graph";

const graph: GraphData = {
  ...emptyGraph(),
  untyped: [
    {
      id: "x_u", source: "a", target: "b", proposed_type: "backs", status: "proposed",
      note_id: "n_1", description: "", text_unit_ids: [],
    },
  ],
  claims: [
    {
      id: "k_1", subject: "a", text: "t", status: "proposed", valid_from: null,
      valid_to: null, supersedes: null, note_id: "n_1", text_unit_ids: [],
    },
  ],
  edges: [
    {
      key: "r:a|supports|b", source: "a", target: "b", type: "supports", proposed_type: null,
      directed: true, weight: 0, traversable: false,
      assertions: [
        {
          id: "x_1", note_id: "n_1", status: "proposed", strength: 5, description: "d",
          direction_corrected: false, text_unit_ids: [],
        },
      ],
    },
  ],
};

describe("foldDecisions", () => {
  it("is last-write-wins and tolerates a torn final line", () => {
    const text = [
      JSON.stringify({ op: "op_1", ts: "t", assertion: "x_1", action: "dismiss", via: "t", reason: null }),
      JSON.stringify({ op: "op_2", ts: "t", assertion: "x_1", action: "confirm", via: "t", reason: null }),
      '{"op": "op_3", "ts": "t", "assertion": "k_1", "act',
    ].join("\n");
    const overlay = foldDecisions(text);
    expect(overlay.get("x_1")).toBe("confirm");
    expect(overlay.has("k_1")).toBe(false);
  });

  it("throws on a malformed interior line", () => {
    const text = '{"bad"\n{"op":"o","ts":"t","assertion":"x","action":"confirm","via":"v","reason":null}\n';
    expect(() => foldDecisions(text)).toThrow(/line 1/);
  });

  it("is empty for an empty file", () => {
    expect(foldDecisions("").size).toBe(0);
  });
});

describe("applyOverlay", () => {
  it("marks statuses and makes an edge traversable once any member is confirmed", () => {
    const overlay = new Map([
      ["x_1", "confirm"],
      ["x_u", "dismiss"],
      ["k_1", "confirm"],
    ] as const);
    const out = applyOverlay(graph, new Map(overlay));
    expect(out.edges[0].assertions[0].status).toBe("confirmed");
    expect(out.edges[0].traversable).toBe(true);
    expect(out.untyped[0].status).toBe("dismissed");
    expect(out.claims[0].status).toBe("confirmed");
    // input untouched
    expect(graph.edges[0].traversable).toBe(false);
    expect(graph.edges[0].assertions[0].status).toBe("proposed");
  });
});

describe("decisionLine", () => {
  it("matches the Python DecisionLog shape with sorted keys", () => {
    const line = decisionLine("x_1", "confirm", null, new Date(Date.UTC(2026, 8, 6, 10, 0, 0)), "abc");
    expect(line).toBe(
      '{"action":"confirm","assertion":"x_1","op":"obsidian_abc","reason":null,"ts":"2026-09-06T10:00:00Z","via":"obsidian"}\n',
    );
  });

  it("round-trips through foldDecisions", () => {
    const line = decisionLine("k_9", "dismiss", "not true", new Date(), "id");
    expect(foldDecisions(line).get("k_9")).toBe("dismiss");
  });
});

describe("applyOverlay, review findings", () => {
  it("never resurrects a superseded claim from its own old confirm", () => {
    const g: GraphData = {
      ...emptyGraph(),
      claims: [{ ...graph.claims[0], id: "k_old", status: "superseded" }],
    };
    const out = applyOverlay(g, new Map([["k_old", "confirm"]]));
    expect(out.claims[0].status).toBe("superseded");
  });

  it("retires the claim a newly confirmed claim supersedes, like the fold", () => {
    const g: GraphData = {
      ...emptyGraph(),
      claims: [
        { ...graph.claims[0], id: "k_old", status: "confirmed" },
        { ...graph.claims[0], id: "k_new", status: "proposed", supersedes: "k_old" },
      ],
    };
    const out = applyOverlay(g, new Map([["k_new", "confirm"]]));
    expect(out.claims.map((c) => c.status)).toEqual(["superseded", "confirmed"]);
    // A proposed correction retires nothing until someone agrees to it.
    expect(applyOverlay(g, new Map()).claims[0].status).toBe("confirmed");
  });

  it("derives traversable and weight from statuses, so a dismissal retracts them", () => {
    const g: GraphData = {
      ...emptyGraph(),
      edges: [{
        ...graph.edges[0], traversable: true, weight: 1,
        assertions: [
          { ...graph.edges[0].assertions[0], id: "x_c", status: "confirmed" },
          { ...graph.edges[0].assertions[0], id: "x_p", status: "proposed" },
        ],
      }],
    };
    const out = applyOverlay(g, new Map([["x_c", "dismiss"]]));
    expect(out.edges[0].traversable).toBe(false);
    expect(out.edges[0].weight).toBe(0);
    const both = applyOverlay(g, new Map([["x_p", "confirm"]]));
    expect(both.edges[0].weight).toBe(2);
  });
});

describe("decisionLine, review findings", () => {
  it("escapes the line separators Python's splitlines would break on", () => {
    const line = decisionLine("x_1", "dismiss", "a\u2028b\u2029c\u0085d", new Date(0), "id");
    expect(line).not.toMatch(/[\u2028\u2029\u0085]/);
    expect(JSON.parse(line).reason).toBe("a\u2028b\u2029c\u0085d");
    expect(line.split("\n").length).toBe(2);
  });
});

describe("conformance with the Python DecisionLog", () => {
  const fixture = readFileSync(resolve(__dirname, "../../tests/fixtures/decisions-conformance.jsonl"), "utf8");

  it("folds the shared fixture to the same map, torn final line included", () => {
    const overlay = foldDecisions(fixture);
    expect([...overlay.entries()]).toEqual([["x_1", "confirm"], ["k_2", "confirm"]]);
  });

  it("treats a torn line before a trailing newline as final, like splitlines()", () => {
    expect(foldDecisions('{"action":"confirm","assertion":"x","op":"o","ts":"t","via":"v"}\n{"bad\n').get("x")).toBe("confirm");
  });

  it("rejects a line missing the keys Python reads", () => {
    expect(() => foldDecisions('{"assertion":"x_1","action":"confirm"}')).toThrow(/missing op, ts, via/);
  });
});
