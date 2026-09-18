import { describe, expect, it } from "vitest";
import { formatTime } from "./format";

describe("formatTime", () => {
  it("formats zero seconds", () => {
    expect(formatTime(0)).toBe("0:00");
  });

  it("formats seconds under a minute", () => {
    expect(formatTime(5)).toBe("0:05");
    expect(formatTime(45)).toBe("0:45");
  });

  it("formats whole minutes", () => {
    expect(formatTime(60)).toBe("1:00");
    expect(formatTime(120)).toBe("2:00");
  });

  it("formats minutes and seconds", () => {
    expect(formatTime(65)).toBe("1:05");
    expect(formatTime(754)).toBe("12:34");
  });

  it("truncates fractional seconds", () => {
    expect(formatTime(5.9)).toBe("0:05");
    expect(formatTime(59.999)).toBe("0:59");
  });

  it("switches to H:MM:SS at one hour", () => {
    expect(formatTime(3600)).toBe("1:00:00");
    expect(formatTime(3661)).toBe("1:01:01");
  });

  it("uses H:MM:SS when long is true", () => {
    expect(formatTime(65, { long: true })).toBe("0:01:05");
    expect(formatTime(3661, { long: true })).toBe("1:01:01");
  });

  it("preserves sub-second precision with decimals option", () => {
    expect(formatTime(5.1, { decimals: 2 })).toBe("0:05.10");
    expect(formatTime(65.85, { decimals: 2 })).toBe("1:05.85");
    expect(formatTime(0, { decimals: 2 })).toBe("0:00.00");
    expect(formatTime(3661.5, { decimals: 2 })).toBe("1:01:01.50");
  });

  it("returns em dash for invalid input", () => {
    expect(formatTime(NaN)).toBe("—");
    expect(formatTime(Infinity)).toBe("—");
    expect(formatTime(-1)).toBe("—");
    expect(formatTime(-Infinity)).toBe("—");
  });
});
