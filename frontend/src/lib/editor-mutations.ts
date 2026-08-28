import { useCallback, useRef, useState } from "react";

import { ApiError, apiFetch } from "./api-client";
import type { Segment } from "../components/TranscriptPlayer";

// -- Pure helpers (no hooks) -------------------------------------------------

export function isTarget(seg: Segment): boolean {
  return seg.reviewTarget && !seg.verified;
}

export function siblingCount(
  segments: Segment[],
  sourceSegmentId: string | null,
): number {
  if (sourceSegmentId === null) return 0;
  return segments.filter((s) => s.sourceSegmentId === sourceSegmentId).length;
}

export function nextTarget(segments: Segment[], from: number): number {
  for (let i = Math.max(from, 0); i < segments.length; i += 1) {
    if (isTarget(segments[i])) return i;
  }
  for (let i = 0; i < Math.min(from, segments.length); i += 1) {
    if (isTarget(segments[i])) return i;
  }
  return -1;
}

// -- Shared mutation hooks ---------------------------------------------------

export interface SegmentPatchResult {
  verified: boolean;
  corrected: boolean;
  text: string;
  progress: { verified: number; total: number };
}

export function useFormPost(
  reviewToken: string | null,
  writable: boolean,
  onClaimLost: () => void,
  onError: (msg: string) => void,
) {
  return useCallback(
    async <T,>(
      path: string,
      body: Record<string, string>,
      opts?: { claimLostOnConflict?: boolean },
    ): Promise<T | null> => {
      if (!writable || reviewToken === null) return null;
      try {
        const res = await apiFetch(path, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
          body: new URLSearchParams({ token: reviewToken, ...body }).toString(),
        });
        return (await res.json()) as T;
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          if (err.conflictKind === "claim") {
            onClaimLost();
          } else if (opts?.claimLostOnConflict === false) {
            onError(err.detail);
          } else {
            onClaimLost();
          }
        } else {
          onError(err instanceof ApiError ? err.detail : "Request failed.");
        }
        return null;
      }
    },
    [writable, reviewToken, onClaimLost, onError],
  );
}

export function useSegmentPatch(
  segments: Segment[],
  setSegments: (s: Segment[]) => void,
  setProgress: (p: { verified: number; total: number }) => void,
) {
  return useCallback(
    (
      index: number,
      result: SegmentPatchResult,
      opts?: { supersedeProvenance?: boolean },
    ): Segment[] => {
      const parentId = segments[index]?.sourceSegmentId ?? null;
      const split = siblingCount(segments, parentId) > 1;
      const patched = segments.map((seg, i) => {
        const isSibling =
          parentId !== null ? seg.sourceSegmentId === parentId : i === index;
        if (!isSibling) return seg;
        return {
          ...seg,
          verified: result.verified,
          corrected: result.corrected,
          text: split ? seg.text : result.text,
          corrections:
            opts?.supersedeProvenance && result.corrected && !split
              ? null
              : seg.corrections,
        };
      });
      setSegments(patched);
      setProgress(result.progress);
      return patched;
    },
    [segments, setSegments, setProgress],
  );
}

export interface WalkCursorState {
  cursor: number;
  setCursor: (i: number) => void;
  goTo: (i: number) => void;
  jumpNext: () => void;
  remaining: number;
}

export function useWalkCursor(
  segments: Segment[],
  initialSegments: Segment[],
  play: (index: number) => void,
): WalkCursorState {
  const [cursor, setCursor] = useState<number>(() =>
    Math.max(nextTarget(initialSegments, 0), 0),
  );

  const goTo = useCallback(
    (index: number) => {
      if (index < 0) return;
      setCursor(index);
      play(index);
    },
    [play],
  );

  const jumpNext = useCallback(() => {
    const next = nextTarget(segments, cursor + 1);
    if (next >= 0) goTo(next);
  }, [segments, cursor, goTo]);

  const remaining = segments.filter(isTarget).length;

  return { cursor, setCursor, goTo, jumpNext, remaining };
}

// Synchronous re-entry guard: state flips a render too late to stop a second
// key that fires before React re-renders, so two writes could overlap.
export function useBusyGuard(): {
  busy: boolean;
  busyRef: React.RefObject<boolean>;
  setBusy: (b: boolean) => void;
} {
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  return { busy, busyRef, setBusy };
}
