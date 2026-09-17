import { useCallback, useEffect, useRef, useState } from "react";

export const INITIAL_CHUNK = 200;
export const APPEND_CHUNK = 100;

export interface UseProgressiveRenderOptions {
  /** Number of segments in the full array. */
  totalCount: number;
  /** Optional index that must always be included (e.g., the cursor). */
  minIndex?: number;
}

export interface UseProgressiveRenderResult {
  /** Exclusive end index: render segments[0..renderedEnd). */
  renderedEnd: number;
  /** Attach to the sentinel div after the last rendered row. */
  sentinelRef: React.RefObject<HTMLDivElement | null>;
  /** Queue expansion to include a target; the DOM updates on React's commit. */
  ensureRendered: (targetIndex: number) => void;
  /** Show every segment. */
  showAll: () => void;
}

export function useProgressiveRender({
  totalCount,
  minIndex,
}: UseProgressiveRenderOptions): UseProgressiveRenderResult {
  const [renderedEnd, setRenderedEnd] = useState(() => {
    const initialEnd = Math.min(INITIAL_CHUNK, totalCount);
    return minIndex != null && minIndex >= initialEnd
      ? Math.min(minIndex + APPEND_CHUNK, totalCount)
      : initialEnd;
  });
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    setRenderedEnd((previous) => {
      const end = Math.min(previous, totalCount);
      return minIndex != null && minIndex >= end
        ? Math.min(minIndex + APPEND_CHUNK, totalCount)
        : end;
    });
  }, [totalCount, minIndex]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || renderedEnd >= totalCount) return;

    // Re-observe after each committed append so an intersecting sentinel keeps
    // filling a tall viewport. Ignore queued callbacks from obsolete observers.
    let appended = false;
    const observer = new IntersectionObserver(
      (entries) => {
        if (
          observerRef.current !== observer ||
          appended ||
          !entries.some((entry) => entry.isIntersecting)
        )
          return;
        appended = true;
        setRenderedEnd((end) => Math.min(end + APPEND_CHUNK, totalCount));
      },
      { root: null, rootMargin: "0px 0px 600px 0px" },
    );
    observerRef.current = observer;
    observer.observe(sentinel);
    return () => {
      observer.disconnect();
      observerRef.current = null;
    };
  }, [renderedEnd, totalCount]);

  const ensureRendered = useCallback(
    (targetIndex: number) => {
      setRenderedEnd((end) =>
        targetIndex < end
          ? end
          : Math.max(end, Math.min(targetIndex + APPEND_CHUNK, totalCount)),
      );
    },
    [totalCount],
  );

  const showAll = useCallback(() => {
    setRenderedEnd(totalCount);
  }, [totalCount]);

  return { renderedEnd, sentinelRef, ensureRendered, showAll };
}
