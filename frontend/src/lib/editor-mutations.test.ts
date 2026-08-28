import { describe, expect, it } from "vitest";

import type { Segment } from "../components/TranscriptPlayer";
import { isTarget, nextTarget, siblingCount } from "./editor-mutations";

function seg(overrides: Partial<Segment> = {}): Segment {
  return {
    start: 0,
    end: 1,
    speaker: "SPEAKER_00",
    text: "hello",
    label: null,
    paletteIndex: null,
    confidence: null,
    segmentId: "s1",
    verified: false,
    corrected: false,
    sourceSegmentId: "s1",
    reviewTarget: true,
    wordStart: null,
    wordEnd: null,
    wordRangeSpeakerId: null,
    corrections: null,
    rawText: null,
    ...overrides,
  };
}

describe("isTarget", () => {
  it("returns true for an unverified review target", () => {
    expect(isTarget(seg())).toBe(true);
  });

  it("returns false when verified", () => {
    expect(isTarget(seg({ verified: true }))).toBe(false);
  });

  it("returns false when not a review target", () => {
    expect(isTarget(seg({ reviewTarget: false }))).toBe(false);
  });
});

describe("siblingCount", () => {
  it("returns 0 for null sourceSegmentId", () => {
    expect(siblingCount([seg()], null)).toBe(0);
  });

  it("counts segments sharing the same sourceSegmentId", () => {
    const segs = [
      seg({ sourceSegmentId: "p1", segmentId: "c1" }),
      seg({ sourceSegmentId: "p1", segmentId: "c2" }),
      seg({ sourceSegmentId: "p2", segmentId: "c3" }),
    ];
    expect(siblingCount(segs, "p1")).toBe(2);
    expect(siblingCount(segs, "p2")).toBe(1);
    expect(siblingCount(segs, "p3")).toBe(0);
  });
});

describe("nextTarget", () => {
  it("finds the next target at or after `from`", () => {
    const segs = [
      seg({ verified: true }),
      seg({ segmentId: "s2", sourceSegmentId: "s2" }),
      seg({ segmentId: "s3", sourceSegmentId: "s3" }),
    ];
    expect(nextTarget(segs, 0)).toBe(1);
    expect(nextTarget(segs, 1)).toBe(1);
    expect(nextTarget(segs, 2)).toBe(2);
  });

  it("wraps around when no target after `from`", () => {
    const segs = [
      seg({ segmentId: "s1", sourceSegmentId: "s1" }),
      seg({ verified: true }),
    ];
    expect(nextTarget(segs, 1)).toBe(0);
  });

  it("returns -1 when no targets exist", () => {
    const segs = [
      seg({ verified: true }),
      seg({ verified: true }),
    ];
    expect(nextTarget(segs, 0)).toBe(-1);
  });

  it("returns -1 for empty segments", () => {
    expect(nextTarget([], 0)).toBe(-1);
  });
});
