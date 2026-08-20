import { describe, expect, it } from "vitest";

import {
  buildQuoteParts,
  captureFormFields,
  codePointToUtf16,
  sliceByCodePoints,
  utf16ToCodePoint,
  type CapturePayload,
} from "./selection";

// "a😀b" — the emoji is one code point but TWO UTF-16 code units (a surrogate
// pair). "é" — e + combining acute is TWO code points (offsets are code
// points, not grapheme clusters, per docs/annotations.md).
const EMOJI = "a\u{1F600}b"; // a, 😀, b  -> 3 code points, 4 code units
const COMBINING = "éx"; // e, ́, x  -> 3 code points, 3 code units

describe("utf16ToCodePoint", () => {
  it("counts an ASCII prefix one-to-one", () => {
    expect(utf16ToCodePoint("hello", 0)).toBe(0);
    expect(utf16ToCodePoint("hello", 3)).toBe(3);
    expect(utf16ToCodePoint("hello", 5)).toBe(5);
  });

  it("counts a surrogate pair as one code point", () => {
    // Before the emoji: 1 unit -> 1 cp. After it: 3 units -> 2 cp. End: 4 -> 3.
    expect(utf16ToCodePoint(EMOJI, 1)).toBe(1);
    expect(utf16ToCodePoint(EMOJI, 3)).toBe(2);
    expect(utf16ToCodePoint(EMOJI, 4)).toBe(3);
  });

  it("treats each combining mark as its own code point", () => {
    expect(utf16ToCodePoint(COMBINING, 2)).toBe(2);
    expect(utf16ToCodePoint(COMBINING, 3)).toBe(3);
  });

  it("clamps out-of-range code-unit offsets", () => {
    expect(utf16ToCodePoint("hi", 99)).toBe(2);
    expect(utf16ToCodePoint("hi", -1)).toBe(0);
  });
});

describe("codePointToUtf16 (inverse of utf16ToCodePoint)", () => {
  it("round-trips across a surrogate pair", () => {
    for (let u16 = 0; u16 <= EMOJI.length; u16 += 1) {
      // Only test unit offsets that fall on a code-point boundary.
      const cp = utf16ToCodePoint(EMOJI, u16);
      if (codePointToUtf16(EMOJI, cp) === u16) {
        expect(utf16ToCodePoint(EMOJI, codePointToUtf16(EMOJI, cp))).toBe(cp);
      }
    }
    // The emoji starts at code point 1 -> code unit 1; after it, cp 2 -> unit 3.
    expect(codePointToUtf16(EMOJI, 1)).toBe(1);
    expect(codePointToUtf16(EMOJI, 2)).toBe(3);
    expect(codePointToUtf16(EMOJI, 3)).toBe(4);
  });

  it("clamps past the end", () => {
    expect(codePointToUtf16("hi", 99)).toBe(2);
  });
});

describe("sliceByCodePoints", () => {
  it("slices an emoji as a single unit", () => {
    expect(sliceByCodePoints(EMOJI, 1, 2)).toBe("\u{1F600}");
    expect(sliceByCodePoints(EMOJI, 0, 1)).toBe("a");
    expect(sliceByCodePoints(EMOJI, 2, 3)).toBe("b");
  });

  it("keeps a combining mark with its base only when both are in range", () => {
    expect(sliceByCodePoints(COMBINING, 0, 2)).toBe("é");
    expect(sliceByCodePoints(COMBINING, 1, 2)).toBe("́");
  });

  it("returns empty for a reversed or out-of-range range", () => {
    expect(sliceByCodePoints("hello", 3, 1)).toBe("");
    expect(sliceByCodePoints("hello", 10, 20)).toBe("");
  });
});

describe("captureFormFields", () => {
  const base: CapturePayload = {
    start: {
      segmentId: "s1",
      offset: 0,
      childWordStart: null,
      childWordEnd: null,
    },
    end: {
      segmentId: "s1",
      offset: 5,
      childWordStart: null,
      childWordEnd: null,
    },
    clientQuote: "hello",
  };

  it("omits null child indices so the server sees them as absent", () => {
    const fields = captureFormFields(base);
    expect(fields).toEqual({
      start_segment_id: "s1",
      start_offset: "0",
      end_segment_id: "s1",
      end_offset: "5",
      client_quote: "hello",
    });
  });

  it("includes child indices when the endpoint sits in a split child", () => {
    const fields = captureFormFields({
      ...base,
      start: { segmentId: "s1", offset: 0, childWordStart: 0, childWordEnd: 2 },
    });
    expect(fields.start_child_word_start).toBe("0");
    expect(fields.start_child_word_end).toBe("2");
  });
});

describe("buildQuoteParts", () => {
  // Mirrors the server's `_derive_quote`: first line's tail + whole middles +
  // last line's head, newline-joined. A byte match is the non-stale signal.
  it("joins a two-line selection as head-tail with no middles", () => {
    expect(buildQuoteParts("say two", 4, [], "three now", 5)).toBe("two\nthree");
  });

  it("includes every whole line between the endpoints (>= 3 segments)", () => {
    expect(
      buildQuoteParts("say two", 4, ["middle words", "and more"], "three now", 5),
    ).toBe("two\nmiddle words\nand more\nthree");
  });

  it("is astral-safe: offsets and slices count code points, not UTF-16 units", () => {
    // "a😀 two": code points a(0) 😀(1) space(2) t(3)... a code-point offset of 2
    // is the space; a UTF-16 reading would land mid-surrogate.
    expect(buildQuoteParts("a\u{1F600} two", 2, [], "end", 3)).toBe(" two\nend");
  });
});
