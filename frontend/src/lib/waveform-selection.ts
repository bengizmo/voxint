import type { TimeRange } from "./peaks";

const DRAG_THRESHOLD = 5;

export { DRAG_THRESHOLD };

export function normalizeRange(
  t1: number,
  t2: number,
  duration: number,
): TimeRange | null {
  if (!Number.isFinite(t1) || !Number.isFinite(t2) || !Number.isFinite(duration) || duration <= 0) return null;
  const clamp = (t: number) => Math.max(0, Math.min(duration, t));
  const start = clamp(Math.min(t1, t2));
  const end = clamp(Math.max(t1, t2));
  if (end <= start) return null;
  return { start, end };
}

export function isDragDistance(startX: number, currentX: number): boolean {
  return Math.abs(currentX - startX) > DRAG_THRESHOLD;
}
