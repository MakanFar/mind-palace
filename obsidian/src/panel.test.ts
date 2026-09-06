import { describe, expect, it } from "vitest";

import { describeValidity, holdsAt } from "./panel";

describe("validity", () => {
  it("describes intervals", () => {
    expect(describeValidity("2015", "2026-03")).toBe("2015 → 2026-03");
    expect(describeValidity("2015", null)).toBe("from 2015");
    expect(describeValidity(null, "unknown")).toBe("until unknown");
    expect(describeValidity(null, null)).toBe("");
  });
  it("holdsAt matches the Python rule on common-prefix comparison", () => {
    expect(holdsAt("2015", "2026-03", "2020-06-01")).toBe(true);
    expect(holdsAt("2015", "2026-03", "2026-04-01")).toBe(false);
    expect(holdsAt("2026-03-15", null, "2026-03")).toBe(true);
    expect(holdsAt(null, "unknown", "1999")).toBe(true);
  });
});
