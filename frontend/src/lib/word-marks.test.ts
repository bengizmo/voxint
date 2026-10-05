import { describe, expect, it } from "vitest";
import { emissionKey, indexEmissions, marksClearedNotice, toggleWordMark, WordMarksRequestGuard,
  type WordMarksPayload, type WordMarkUnit } from "./word-marks";

const unit: WordMarkUnit = { start: 1, end: 2, from: 2, to: 4, removed: null, protected: false, mark: null };
const payload: WordMarksPayload = { runId: "run", version: "1.hash", fillerListDefault: true, detectionError: null, emissions: [], stale: [] };

describe("word mark toggles", () => {
  it.each([
    [{ ...unit, mark: "keep" as const }, "f", "clear"],
    [{ ...unit, removed: "filler" as const }, "f", "keep"],
    [{ ...unit, removed: "phrase" as const }, "f", "keep"],
    [{ ...unit, protected: true }, "f", "keep"],
    [{ ...unit, mark: "omit" as const }, "o", "clear"],
    [unit, "o", "omit"],
  ] as const)("toggles %o via %s", (target, key, action) => {
    expect(toggleWordMark(target, key, null)).toEqual({ action });
  });
  it("refuses ordinary keep and broken detection while allowing clear/omit", () => {
    expect(toggleWordMark(unit, "f", null).error).toBe("Only a word the filler list removes can be kept.");
    expect(toggleWordMark({ ...unit, removed: "filler" }, "f", "broken").error).toBe("Filler detection is unavailable, so no word can be kept right now.");
    expect(toggleWordMark(unit, "f", "broken").error).toBe("Filler detection is unavailable, so no word can be kept right now.");
    expect(toggleWordMark({ ...unit, mark: "keep" }, "f", "broken").action).toBe("clear");
    expect(toggleWordMark(unit, "o", "broken").action).toBe("omit");
  });
});
it("keys unsplit null separately from child token zero", () => {
  const base = { segmentId: "seg", wordEnd: null, markable: true, reason: null, units: [] };
  const index = indexEmissions({ ...payload, emissions: [{ ...base, wordStart: null }, { ...base, wordStart: 0 }] });
  expect(index.get(emissionKey("seg", null))?.wordStart).toBeNull();
  expect(index.get(emissionKey("seg", 0))?.wordStart).toBe(0);
});
it("drops superseded responses and skips identical versions only at the same focus", () => {
  const guard = new WordMarksRequestGuard();
  const old = guard.begin();
  const newest = guard.begin();
  expect(guard.accept(old, payload, null)).toBe(false);
  expect(guard.accept(newest, payload, null)).toBe(true);
  expect(guard.accept(guard.begin(), payload, null)).toBe(false);
  expect(guard.accept(guard.begin(), payload, "seg")).toBe(true);
  const pending = guard.begin();
  guard.invalidate();
  expect(guard.accept(pending, { ...payload, version: "2.hash" }, "seg")).toBe(false);
  expect(guard.accept(guard.begin(), payload, "seg")).toBe(true);
});
it("announces cleared marks with singular/plural and no zero notice", () => {
  expect(marksClearedNotice(0)).toBeNull();
  expect(marksClearedNotice(1)).toBe("1 clean-up mark was cleared because the text changed.");
  expect(marksClearedNotice(2)).toBe("2 clean-up marks were cleared because the text changed.");
});
