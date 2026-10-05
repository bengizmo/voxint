import type { Segment } from "../components/TranscriptPlayer";

export interface WordMarkUnit {
  start: number;
  end: number;
  from: number;
  to: number;
  removed: "filler" | "phrase" | "omit" | null;
  protected: boolean;
  mark: "keep" | "omit" | null;
}
export interface WordMarkEmission {
  segmentId: string;
  wordStart: number | null;
  wordEnd: number | null;
  markable: boolean;
  reason: string | null;
  units: WordMarkUnit[];
}
export interface StaleWordMark {
  segmentId: string;
  start: number;
  end: number;
  action: "keep" | "omit";
  segmentStart: number;
}
export interface WordMarkUndo {
  kind: "word-mark";
  markId: string;
  expiresAt: string;
}
export interface WordMarksPayload {
  runId: string;
  fillerListDefault: boolean | null;
  detectionError: string | null;
  emissions: WordMarkEmission[];
  stale: StaleWordMark[];
  version: string;
}
export interface WordMarksResult extends WordMarksPayload {
  undo: WordMarkUndo | null;
}
export type WordMarkAction = "keep" | "omit" | "clear";

// word_marks_view.py emits null for an unsplit parent, NOT token index zero.
// Split children carry child.word_start, matching the island's wordStart exactly.
export function emissionKey(segmentId: string | null, wordStart: number | null): string {
  return JSON.stringify([segmentId, wordStart]);
}
export function indexEmissions(payload: WordMarksPayload): Map<string, WordMarkEmission> {
  return new Map(payload.emissions.map((e) => [emissionKey(e.segmentId, e.wordStart), e]));
}
export function wordMarksByLine(segments: Segment[], payload: WordMarksPayload): Map<number, WordMarkEmission> {
  const index = indexEmissions(payload);
  const lines = new Map<number, WordMarkEmission>();
  segments.forEach((seg, i) => {
    const emission = index.get(emissionKey(seg.sourceSegmentId, seg.wordStart));
    if (emission) lines.set(i, emission);
  });
  return lines;
}
export function staleMarksByLine(segments: Segment[], payload: WordMarksPayload): Map<number, StaleWordMark[]> {
  const lines = new Map<number, StaleWordMark[]>();
  for (const mark of payload.stale) {
    // One notice per parent, even when its stale range no longer fits a child.
    const i = segments.findIndex((s) => s.sourceSegmentId === mark.segmentId);
    if (i >= 0) lines.set(i, [...(lines.get(i) ?? []), mark]);
  }
  return lines;
}
export function wordMarkStyle(unit: WordMarkUnit): { className: string; title: string } | null {
  if (unit.mark === "omit") return { className: "wm-omit", title: "omitted" };
  if (unit.mark === "keep") return { className: "wm-keep", title: "kept" };
  if (unit.removed === "filler" || unit.removed === "phrase") {
    return { className: "wm-removed", title: "removed by filler clean-up" };
  }
  return null;
}
export function toggleWordMark(unit: WordMarkUnit, key: "f" | "o", detectionError: string | null):
  { action: WordMarkAction; error?: never } | { action?: never; error: string } {
  if (key === "o") return { action: unit.mark === "omit" ? "clear" : "omit" };
  if (unit.mark === "keep") return { action: "clear" };
  if (detectionError) return { error: "Filler detection is unavailable, so no word can be kept right now." };
  if (unit.removed === "filler" || unit.removed === "phrase" || unit.protected) return { action: "keep" };
  return { error: "Only a word the filler list removes can be kept." };
}
export function marksClearedNotice(count: number): string | null {
  if (count <= 0) return null;
  return `${count} clean-up ${count === 1 ? "mark was" : "marks were"} cleared because the text changed.`;
}

// Request order, not lexicographic version order, governs adoption. Versions are
// opaque state hashes and focus-independent; the same version at a NEW focus
// must still land because that response includes additional ordinary units.
export class WordMarksRequestGuard {
  private latest = 0;
  private current: { version: string; focus: string | null } | null = null;
  begin(): number { return ++this.latest; }
  invalidate(): void { this.latest += 1; this.current = null; }
  isLatest(request: number): boolean { return request === this.latest; }
  accept(request: number, payload: WordMarksPayload, focus: string | null): boolean {
    if (request !== this.latest) return false;
    if (this.current?.version === payload.version && this.current.focus === focus) return false;
    this.current = { version: payload.version, focus };
    return true;
  }
}
