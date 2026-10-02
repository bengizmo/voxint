import { type RefObject, useCallback, useRef, useState } from "react";

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
        // Every review write marks a lost claim (#728); any other 409 is a
        // state conflict the operator can read and act on.
        if (err instanceof ApiError && err.conflictKind === "claim") {
          onClaimLost();
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
  goTo: (i: number) => boolean;
  select: (i: number) => boolean;
  jumpNext: () => boolean;
  remaining: number;
}

// `canLeave` is asked before the cursor leaves its line; returning false keeps
// it there (the editor warns about an unsaved edit first, issue #732). goTo,
// select and jumpNext report whether the cursor went. setCursor is unguarded,
// for moves that already confirmed.
export function useWalkCursor(
  segments: Segment[],
  initialSegments: Segment[],
  play: (index: number) => void,
  canLeave: () => boolean = () => true,
): WalkCursorState {
  const [cursor, setCursor] = useState<number>(() =>
    Math.max(nextTarget(initialSegments, 0), 0),
  );

  const select = useCallback(
    (index: number): boolean => {
      if (index < 0) return false;
      if (index !== cursor && !canLeave()) return false;
      setCursor(index);
      return true;
    },
    [cursor, canLeave],
  );

  const goTo = useCallback(
    (index: number): boolean => {
      if (!select(index)) return false;
      play(index);
      return true;
    },
    [select, play],
  );

  const jumpNext = useCallback((): boolean => {
    const next = nextTarget(segments, cursor + 1);
    return next >= 0 && goTo(next);
  }, [segments, cursor, goTo]);

  const remaining = segments.filter(isTarget).length;

  return { cursor, setCursor, goTo, select, jumpNext, remaining };
}

export interface WriteGuard {
  busy: boolean;
  busyRef: RefObject<boolean>;
  setBusy: (busy: boolean) => void;
}

// Synchronous re-entry guard: state flips a render too late to stop a second
// key that fires before React re-renders, so two writes could overlap.
export function useBusyGuard(): WriteGuard {
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  return { busy, busyRef, setBusy };
}
