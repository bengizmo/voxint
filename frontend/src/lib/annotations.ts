// The operator annotation layer's wire shapes and pure view helpers (issue #86).
//
// These interfaces mirror the server's island shape EXACTLY — the keys are pinned
// by tests/contracts/test_annotation_island_contract.py, so a drift here rots the
// console silently. The helpers are pure (no DOM, no React) and unit-tested in
// annotations.test.ts; the React state and network I/O live in
// components/AnnotationLayer.tsx.

// One highlight painted on a rendered line: a half-open [start, end) code-point
// range into the line's text and its fixed-palette color index. TranscriptPlayer
// renders these as <mark class="hl-N"> without altering a single character.
export interface AnnotationLineSpan {
  start: number;
  end: number;
  colorIndex: number;
}

// A global tag as the island sees it. `archived` is a bool the picker uses to hide
// it while a row still carrying it renders it.
export interface AnnotationTagShape {
  id: string;
  name: string;
  color: number;
  archived: boolean;
}

// One resolved span of an annotation against the current corrected render.
export interface AnnotationSpanShape {
  lineIndex: number;
  start: number;
  end: number;
}

// One annotation as the island sees it (the GET/POST/PATCH response body). Speaker
// attribution and timing come from the read-time resolution (always current);
// `quote` is the ORIGINAL captured quote, shown verbatim even when stale. A
// stale annotation carries no `spans` (its text moved) but keeps `locatorLineIndex`
// as an approximate "it was near here" pointer.
export interface AnnotationShape {
  id: string;
  anchorKind: string;
  colorIndex: number;
  quote: string;
  note: string | null;
  operator: string;
  stale: boolean;
  // The server emits exactly one of these (annotations.py TIMING_WORD/SEGMENT):
  // "word" is precise (word-timed), "segment" is approximate (whole-segment bounds).
  timingPrecision: "word" | "segment";
  startSeconds: number | null;
  endSeconds: number | null;
  speakers: string[];
  spans: AnnotationSpanShape[];
  locatorLineIndex: number | null;
  startSegmentIndex: number;
  endSegmentIndex: number;
  tags: AnnotationTagShape[];
}

// The server-enforced caps the toolbar mirrors so it can refuse locally before a
// doomed round-trip (the server still re-checks; these are UX, not the gate).
export interface AnnotationLimits {
  paletteSize: number;
  maxSpanSegments: number;
  maxNoteChars: number;
  maxTagsPerAnnotation: number;
  maxQuoteChars: number;
  maxTagNameChars: number;
}

// A defensive fallback mirroring the server caps (src/voxint/db/models.py). The
// review route always sends the real `annotationLimits`, so this is used only if a
// props payload omits them — never in normal operation.
export const FALLBACK_ANNOTATION_LIMITS: AnnotationLimits = {
  paletteSize: 6,
  maxSpanSegments: 100,
  maxNoteChars: 4000,
  maxTagsPerAnnotation: 8,
  maxQuoteChars: 50000,
  maxTagNameChars: 64,
};

// Group every non-stale annotation's resolved spans by the line they fall on, so
// TranscriptPlayer can paint a line from one lookup. A stale annotation is skipped
// entirely — its text moved, so it gets a locator chip (below), never inline marks.
export function spansByLine(
  annotations: AnnotationShape[],
): Map<number, AnnotationLineSpan[]> {
  const map = new Map<number, AnnotationLineSpan[]>();
  for (const a of annotations) {
    if (a.stale) continue;
    for (const s of a.spans) {
      const arr = map.get(s.lineIndex);
      const span = { start: s.start, end: s.end, colorIndex: a.colorIndex };
      if (arr) arr.push(span);
      else map.set(s.lineIndex, [span]);
    }
  }
  return map;
}

// The set of line indices carrying a stale annotation's approximate locator chip.
export function staleLocatorLines(annotations: AnnotationShape[]): Set<number> {
  const set = new Set<number>();
  for (const a of annotations) {
    if (a.stale && a.locatorLineIndex !== null) set.add(a.locatorLineIndex);
  }
  return set;
}

// A live annotation with no painted line and no locator sorts last. Mirrors the
// server's _ORDER_LINE_SENTINEL so client and server agree on the tail.
const ORDER_LINE_SENTINEL = 1_000_000_000;

function orderLine(a: AnnotationShape): number {
  if (a.spans.length > 0) return a.spans[0].lineIndex;
  return a.locatorLineIndex ?? ORDER_LINE_SENTINEL;
}

// Transcript order for the Highlights panel: by RENDERED line index (not the
// captured start segment index, which a split child reorders against), then the
// first span's offset, then the id as a stable final tiebreak. This is the exact
// key the server's `resolved_order_key` applies, so the panel and the bulk
// pull-quote export can never disagree on order (issue #86).
export function sortAnnotations(
  annotations: AnnotationShape[],
): AnnotationShape[] {
  return [...annotations].sort((a, b) => {
    const al = orderLine(a);
    const bl = orderLine(b);
    if (al !== bl) return al - bl;
    const as = a.spans[0]?.start ?? 0;
    const bs = b.spans[0]?.start ?? 0;
    if (as !== bs) return as - bs;
    return a.id.localeCompare(b.id);
  });
}

// The bulk pull-quote export URL for the current OR-union tag filter (issue #86):
// each selected tag becomes a repeated `?tag=` param, exactly the filter the panel
// applies, so a "Copy all (filtered)" fetch returns the same set the operator sees.
// An empty filter yields the bare route (all highlights).
export function annotationsExportUrl(runId: string, tagIds: Set<string>): string {
  const params = new URLSearchParams();
  for (const t of tagIds) params.append("tag", t);
  const qs = params.toString();
  return `/review/${runId}/annotations/export.md${qs ? `?${qs}` : ""}`;
}

// OR-union tag filter for the panel: an empty filter keeps everything; otherwise a
// row survives when it carries at least one of the selected tags.
export function filterByTags(
  annotations: AnnotationShape[],
  tagIds: Set<string>,
): AnnotationShape[] {
  if (tagIds.size === 0) return annotations;
  return annotations.filter((a) => a.tags.some((t) => tagIds.has(t.id)));
}
