import { describe, expect, it } from "vitest";

import { parseJumpParam, resolveJumpIndex } from "./jump";

describe("parseJumpParam", () => {
  it("returns null when ?t is absent", () => {
    expect(parseJumpParam("")).toBeNull();
    expect(parseJumpParam("?q=hello")).toBeNull();
  });

  it("returns null for blank or non-numeric values", () => {
    expect(parseJumpParam("?t=")).toBeNull();
    expect(parseJumpParam("?t=%20")).toBeNull();
    expect(parseJumpParam("?t=abc")).toBeNull();
  });

  it("returns null for a negative time", () => {
    expect(parseJumpParam("?t=-5")).toBeNull();
  });

  it("parses zero and whole seconds", () => {
    expect(parseJumpParam("?t=0")).toBe(0);
    expect(parseJumpParam("?t=42")).toBe(42);
  });

  it("preserves fractional seconds", () => {
    expect(parseJumpParam("?t=12.5")).toBe(12.5);
  });

  it("ignores other params around t", () => {
    expect(parseJumpParam("?q=hi&t=7&x=1")).toBe(7);
  });

  it("returns null for a non-finite time", () => {
    expect(parseJumpParam("?t=Infinity")).toBeNull();
    expect(parseJumpParam("?t=1e309")).toBeNull();
    expect(parseJumpParam("?t=NaN")).toBeNull();
  });
});

describe("resolveJumpIndex", () => {
  const segs = [
    { start: 0, end: 5 },
    { start: 5, end: 10 },
    // a gap (silence) between 10 and 20
    { start: 20, end: 30 },
  ];

  it("finds the line whose span contains t", () => {
    expect(resolveJumpIndex(segs, 0)).toBe(0);
    expect(resolveJumpIndex(segs, 4.9)).toBe(0);
    expect(resolveJumpIndex(segs, 5)).toBe(1);
    expect(resolveJumpIndex(segs, 25)).toBe(2);
  });

  it("snaps a time in a gap forward to the next line", () => {
    expect(resolveJumpIndex(segs, 12)).toBe(2);
  });

  it("returns -1 when t is past the last line", () => {
    expect(resolveJumpIndex(segs, 999)).toBe(-1);
  });

  it("returns -1 for an empty transcript", () => {
    expect(resolveJumpIndex([], 3)).toBe(-1);
  });

  it("snaps t at a line's exclusive end forward (half-open [start, end))", () => {
    // 10 is the end of line 1 [5, 10) and not contained by it; it snaps to the
    // next line starting at or after 10 (here the gap-forward line 2).
    expect(resolveJumpIndex(segs, 10)).toBe(2);
  });

  it("lands the fractional start on its own line, not the previous one", () => {
    // The bug the fractional ?t= fix guards: with contiguous lines, a target
    // start of 10.9 truncated to 10 would resolve into the previous [9.2, 10.9)
    // line. Passing the true fractional start resolves to the target line.
    const contiguous = [
      { start: 9.2, end: 10.9 },
      { start: 10.9, end: 12.0 },
    ];
    expect(resolveJumpIndex(contiguous, 10.9)).toBe(1);
    expect(resolveJumpIndex(contiguous, 10)).toBe(0); // documents the truncation trap
  });
});
