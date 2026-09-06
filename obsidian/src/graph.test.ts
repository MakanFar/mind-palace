import { describe, expect, it } from "vitest";

import { emptyGraph, parseGraph } from "./graph";

describe("parseGraph", () => {
  it("accepts version 1 and rejects others", () => {
    const ok = parseGraph(JSON.stringify(emptyGraph()));
    expect(ok.entities).toEqual([]);
    expect(() => parseGraph(JSON.stringify({ version: 2 }))).toThrow(/version 2/);
  });
});
