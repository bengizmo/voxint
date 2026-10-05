import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiFetch } from "./api-client";
import { WordMarksRequestGuard, type WordMarksPayload } from "./word-marks";

export function useWordMarks(runId: string, onError: (message: string) => void) {
  // Each run has its own terminal scheduler, so old write callbacks cannot
  // revive reads or adopt a payload into the next run.
  const scheduler = useMemo(() => ({
    runId, guard: new WordMarksRequestGuard(), desiredFocus: null as string | null,
    writePending: false, refreshPending: false, disposed: false,
  }), [runId]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const loadErrorRef = useRef<string | null>(null);
  const [snapshot, setSnapshot] = useState<{ payload: WordMarksPayload; focus: string | null } | null>(null);
  const refresh = useCallback(async (focus = scheduler.desiredFocus, signal?: AbortSignal) => {
    if (scheduler.disposed) return;
    scheduler.desiredFocus = focus;
    // This flag changes synchronously at write start, independently of React's
    // busy renders. Coalesce focus changes and explicit refreshes until the
    // response has been adopted (or the write has failed).
    if (scheduler.writePending) {
      scheduler.refreshPending = true;
      return;
    }
    const request = scheduler.guard.begin();
    try {
      const res = await apiFetch(`/review/${scheduler.runId}/word-marks${focus === null ? "" : `?segment=${encodeURIComponent(focus)}`}`, {
        headers: { accept: "application/json" }, cache: "no-store", signal,
      });
      const payload = await res.json() as WordMarksPayload;
      if (scheduler.guard.isLatest(request) && loadErrorRef.current !== null) {
        loadErrorRef.current = null;
        setLoadError(null);
      }
      if (scheduler.guard.accept(request, payload, focus)) setSnapshot({ payload, focus });
    } catch (err) {
      // Drop failures from superseded reads too.
      if (scheduler.guard.isLatest(request)) {
        const message = err instanceof ApiError ? err.detail : "Could not load clean-up marks.";
        loadErrorRef.current = message;
        setLoadError(message);
        onError(message);
      }
    }
  }, [onError, scheduler]);
  const adopt = useCallback((payload: WordMarksPayload, focus: string | null, request = scheduler.guard.begin()) => {
    if (scheduler.disposed) return;
    if (scheduler.guard.accept(request, payload, focus)) setSnapshot({ payload, focus });
    if (scheduler.writePending) {
      // The write is authoritative at its own focus. Returning to that focus
      // during the POST needs no extra read; a different focus needs one.
      scheduler.refreshPending = !scheduler.guard.isLatest(request) || scheduler.desiredFocus !== focus;
    } else if (!scheduler.guard.isLatest(request) || scheduler.desiredFocus !== focus) void refresh();
  }, [refresh, scheduler]);
  const requestFocus = useCallback((focus: string | null) => {
    if (scheduler.desiredFocus !== focus) void refresh(focus);
  }, [refresh, scheduler]);
  const beginWrite = useCallback(() => {
    if (scheduler.disposed) return scheduler.guard.begin();
    scheduler.writePending = true;
    return scheduler.guard.begin();
  }, [scheduler]);
  const finishWrite = useCallback((success: boolean) => {
    if (scheduler.disposed) return;
    scheduler.writePending = false;
    if (!success || scheduler.refreshPending) {
      scheduler.refreshPending = false;
      void refresh();
    }
  }, [refresh, scheduler]);
  useEffect(() => {
    scheduler.disposed = false;
    scheduler.writePending = false;
    scheduler.refreshPending = false;
    loadErrorRef.current = null;
    setLoadError(null);
    setSnapshot(null);
    void refresh();
    return () => {
      scheduler.disposed = true;
      scheduler.guard.invalidate();
    };
  }, [refresh, scheduler]);
  return { snapshot, refresh, requestFocus, adopt, beginWrite, finishWrite, loadError };
}
