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
