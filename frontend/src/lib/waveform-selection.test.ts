import { describe, expect, it } from "vitest";
import {
  DRAG_THRESHOLD,
  isDragDistance,
  normalizeRange,
} from "./waveform-selection";

describe("normalizeRange", () => {
  it("returns a range with start < end when t1 < t2", () => {
    expect(normalizeRange(2, 5, 10)).toEqual({ start: 2, end: 5 });
  });

  it("normalizes reversed endpoints (t1 > t2)", () => {
    expect(normalizeRange(7, 3, 10)).toEqual({ start: 3, end: 7 });
  });

  it("clamps start below zero", () => {
    expect(normalizeRange(-1, 3, 10)).toEqual({ start: 0, end: 3 });
  });

  it("clamps end beyond duration", () => {
    expect(normalizeRange(2, 15, 10)).toEqual({ start: 2, end: 10 });
  });

  it("clamps both endpoints", () => {
    expect(normalizeRange(-5, 20, 10)).toEqual({ start: 0, end: 10 });
  });

  it("returns null for zero-width range (same time)", () => {
    expect(normalizeRange(5, 5, 10)).toBeNull();
  });

  it("returns null for non-finite t1", () => {
    expect(normalizeRange(NaN, 5, 10)).toBeNull();
    expect(normalizeRange(Infinity, 5, 10)).toBeNull();
  });

  it("returns null for non-finite t2", () => {
    expect(normalizeRange(2, NaN, 10)).toBeNull();
    expect(normalizeRange(2, -Infinity, 10)).toBeNull();
  });

  it("returns null for zero, negative, or non-finite duration", () => {
    expect(normalizeRange(2, 5, 0)).toBeNull();
    expect(normalizeRange(2, 5, -1)).toBeNull();
    expect(normalizeRange(2, 5, NaN)).toBeNull();
    expect(normalizeRange(2, 5, Infinity)).toBeNull();
  });

  it("handles very small ranges", () => {
    const r = normalizeRange(1.001, 1.002, 60);
    expect(r).not.toBeNull();
    expect(r!.start).toBeCloseTo(1.001);
    expect(r!.end).toBeCloseTo(1.002);
  });

  it("handles fractional seconds", () => {
    expect(normalizeRange(1.5, 3.7, 10)).toEqual({ start: 1.5, end: 3.7 });
  });
});

describe("isDragDistance", () => {
  it("returns false for no movement", () => {
    expect(isDragDistance(100, 100)).toBe(false);
  });

  it("returns false for movement at the threshold", () => {
    expect(isDragDistance(100, 100 + DRAG_THRESHOLD)).toBe(false);
  });

  it("returns true for movement beyond the threshold", () => {
    expect(isDragDistance(100, 100 + DRAG_THRESHOLD + 1)).toBe(true);
  });

  it("works for leftward movement", () => {
    expect(isDragDistance(100, 100 - DRAG_THRESHOLD - 1)).toBe(true);
  });

  it("returns false for subpixel movement", () => {
    expect(isDragDistance(100, 102)).toBe(false);
  });
});
