import { describe, expect, it } from "vitest";

import { linkShown } from "./canvas";

describe("linkShown", () => {
  const confirmed = { kind: "confirmed", pending: false } as const;
  const confirmedWithPending = { kind: "confirmed", pending: true } as const;
  const proposed = { kind: "proposed", pending: true } as const;
  const untyped = { kind: "untyped", pending: true } as const;
  const untypedDecided = { kind: "untyped", pending: false } as const;

  it("all shows every live link", () => {
    for (const link of [confirmed, confirmedWithPending, proposed, untyped, untypedDecided]) {
      expect(linkShown("all", link)).toBe(true);
    }
  });
  it("confirmed shows only traversable edges", () => {
    expect(linkShown("confirmed", confirmed)).toBe(true);
    expect(linkShown("confirmed", confirmedWithPending)).toBe(true);
    expect(linkShown("confirmed", proposed)).toBe(false);
    expect(linkShown("confirmed", untyped)).toBe(false);
  });
  it("proposed shows only links that still have something to decide", () => {
    expect(linkShown("proposed", proposed)).toBe(true);
    expect(linkShown("proposed", untyped)).toBe(true);
    // A traversable edge with one proposed member is still on the review queue.
    expect(linkShown("proposed", confirmedWithPending)).toBe(true);
    expect(linkShown("proposed", confirmed)).toBe(false);
    expect(linkShown("proposed", untypedDecided)).toBe(false);
  });
});
