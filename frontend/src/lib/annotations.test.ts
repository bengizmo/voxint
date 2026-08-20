import { describe, expect, it } from "vitest";

import {
  annotationsExportUrl,
  filterByTags,
  sortAnnotations,
  spansByLine,
  staleLocatorLines,
  type AnnotationShape,
} from "./annotations";

// A minimal annotation with sensible defaults; each test overrides what it needs.
function annotation(over: Partial<AnnotationShape>): AnnotationShape {
  return {
    id: "a",
    anchorKind: "text_range",
    colorIndex: 0,
    quote: "q",
    note: null,
    operator: "op",
    stale: false,
    timingPrecision: "word",
    startSeconds: 0,
    endSeconds: 1,
    speakers: [],
    spans: [],
    locatorLineIndex: null,
    startSegmentIndex: 0,
    endSegmentIndex: 0,
    tags: [],
    ...over,
  };
}

describe("spansByLine", () => {
  it("groups each non-stale annotation's spans under their line, carrying color", () => {
    const map = spansByLine([
      annotation({
        id: "a",
        colorIndex: 2,
        spans: [
          { lineIndex: 0, start: 1, end: 4 },
          { lineIndex: 1, start: 0, end: 3 },
        ],
      }),
      annotation({
        id: "b",
        colorIndex: 5,
        spans: [{ lineIndex: 0, start: 6, end: 9 }],
      }),
    ]);
    expect(map.get(0)).toEqual([
      { start: 1, end: 4, colorIndex: 2 },
      { start: 6, end: 9, colorIndex: 5 },
    ]);
    expect(map.get(1)).toEqual([{ start: 0, end: 3, colorIndex: 2 }]);
  });

  it("omits stale annotations entirely (their text moved)", () => {
    const map = spansByLine([
      annotation({
        id: "a",
        stale: true,
        spans: [{ lineIndex: 0, start: 0, end: 3 }],
      }),
    ]);
    expect(map.size).toBe(0);
  });
});

describe("staleLocatorLines", () => {
  it("collects stale annotations' locator lines and ignores fresh ones", () => {
    const set = staleLocatorLines([
      annotation({ id: "a", stale: true, locatorLineIndex: 4 }),
      annotation({ id: "b", stale: true, locatorLineIndex: null }),
      annotation({ id: "c", stale: false, locatorLineIndex: 7 }),
    ]);
    expect([...set]).toEqual([4]);
  });
});

describe("sortAnnotations", () => {
  it("orders by rendered line index, then first-span offset, then id", () => {
    const order = sortAnnotations([
      annotation({
        id: "late",
        startSegmentIndex: 2,
        spans: [{ lineIndex: 2, start: 0, end: 1 }],
      }),
      annotation({
        id: "early-b",
        startSegmentIndex: 0,
        spans: [{ lineIndex: 0, start: 5, end: 6 }],
      }),
      annotation({
        id: "early-a",
        startSegmentIndex: 0,
        spans: [{ lineIndex: 0, start: 1, end: 2 }],
      }),
    ]).map((a) => a.id);
    expect(order).toEqual(["early-a", "early-b", "late"]);
  });

  it("sorts by rendered line, not captured segment index (split child)", () => {
    // A split child moves a low-segment-index highlight to a later line: the
    // panel must follow the line the operator sees, matching the server key.
    const order = sortAnnotations([
      annotation({
        id: "seg0-line3",
        startSegmentIndex: 0,
        spans: [{ lineIndex: 3, start: 0, end: 1 }],
      }),
      annotation({
        id: "seg5-line1",
        startSegmentIndex: 5,
        spans: [{ lineIndex: 1, start: 0, end: 1 }],
      }),
    ]).map((a) => a.id);
    expect(order).toEqual(["seg5-line1", "seg0-line3"]);
  });

  it("orders a stale row by its locator line and puts an unresolvable row last", () => {
    const order = sortAnnotations([
      annotation({ id: "unresolvable", stale: true, spans: [], locatorLineIndex: null }),
      annotation({ id: "stale-line1", stale: true, spans: [], locatorLineIndex: 1 }),
      annotation({ id: "live-line0", spans: [{ lineIndex: 0, start: 0, end: 1 }] }),
    ]).map((a) => a.id);
    expect(order).toEqual(["live-line0", "stale-line1", "unresolvable"]);
  });

  it("does not mutate the input array", () => {
    const input = [
      annotation({ id: "b", startSegmentIndex: 1 }),
      annotation({ id: "a", startSegmentIndex: 0 }),
    ];
    sortAnnotations(input);
    expect(input.map((a) => a.id)).toEqual(["b", "a"]);
  });
});

describe("filterByTags", () => {
  const rows = [
    annotation({
      id: "a",
      tags: [{ id: "t1", name: "one", color: 0, archived: false }],
    }),
    annotation({
      id: "b",
      tags: [{ id: "t2", name: "two", color: 1, archived: false }],
    }),
    annotation({ id: "c", tags: [] }),
  ];

  it("keeps everything when the filter is empty", () => {
    expect(filterByTags(rows, new Set()).map((a) => a.id)).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("is an OR-union across the selected tags", () => {
    expect(filterByTags(rows, new Set(["t1", "t2"])).map((a) => a.id)).toEqual([
      "a",
      "b",
    ]);
  });
});

describe("annotationsExportUrl", () => {
  it("is the bare route when no tag filter is active", () => {
    expect(annotationsExportUrl("run-1", new Set())).toBe(
      "/review/run-1/annotations/export.md",
    );
  });

  it("appends a repeated ?tag= param per selected tag (OR-union)", () => {
    const url = annotationsExportUrl("run-1", new Set(["t1", "t2"]));
    expect(url.startsWith("/review/run-1/annotations/export.md?")).toBe(true);
    const params = new URLSearchParams(url.split("?")[1]);
    expect(params.getAll("tag").sort()).toEqual(["t1", "t2"]);
  });
});
