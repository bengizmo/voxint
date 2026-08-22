// Deep-link jump target (issue #121). A Meaning search result links to the
// transcript at the passage start via ?t=SECONDS; on load the island scrolls that
// line into view and briefly flashes it. These helpers are pure so the behavior is
// unit-testable without a DOM. No audio seek is involved: a jump is a reading act.

// Parse the ?t= seconds from a location.search string. Absent, blank, non-numeric,
// or negative yields null, which the island treats as "no jump" (a plain transcript
// open, or a hand-edited URL). Fractional seconds are preserved; the caller matches
// them against segment spans.
export function parseJumpParam(search: string): number | null {
  const raw = new URLSearchParams(search).get("t");
  if (raw == null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

// Resolve the transcript line index a jump time lands on. First the line whose
// half-open [start, end) contains t; else the first line starting at or after t (a
// passage start can fall in a gap between segments, e.g. a silence, so it snaps
// forward to the next spoken line); else -1 (t is past the last line, or there are
// none) so the caller no-ops.
export function resolveJumpIndex(
  segments: readonly { start: number; end: number }[],
  t: number,
): number {
  const containing = segments.findIndex((s) => t >= s.start && t < s.end);
  if (containing !== -1) return containing;
  return segments.findIndex((s) => s.start >= t);
}
