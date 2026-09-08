import { describe, expect, it } from "vitest";

import { colourFor, radiusFor } from "./palette";

describe("palette", () => {
  it("has a colour for every configured type and a fallback", () => {
    expect(colourFor("concept")).toBe("#3d84a8");
    expect(colourFor("unknown")).toBe("#8a8a8a");
    expect(colourFor("organisation")).toBe("#6c757d");
  });
  it("sizes by rank with a floor and a cap", () => {
    expect(radiusFor(0)).toBe(4);
    expect(radiusFor(4)).toBe(8);
    expect(radiusFor(1000)).toBe(18);
  });
});
