import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiFetch } from "./api-client";
import { WordMarksRequestGuard, type WordMarksPayload } from "./word-marks";

export function useWordMarks(runId: string, onError: (message: string) => void) {
  const guard = useRef(new WordMarksRequestGuard());
  const desiredFocus = useRef<string | null>(null);
  const writePending = useRef(false);
  const refreshPending = useRef(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const loadErrorRef = useRef<string | null>(null);
  const [snapshot, setSnapshot] = useState<{ payload: WordMarksPayload; focus: string | null } | null>(null);
  const refresh = useCallback(async (focus = desiredFocus.current, signal?: AbortSignal) => {
    desiredFocus.current = focus;
    // This ref changes synchronously at write start, independently of React's
    // busy renders. Coalesce focus changes and explicit refreshes until the
    // response has been adopted (or the write has failed).
    if (writePending.current) {
      refreshPending.current = true;
      return;
    }
    const request = guard.current.begin();
    try {
      const res = await apiFetch(`/review/${runId}/word-marks${focus === null ? "" : `?segment=${encodeURIComponent(focus)}`}`, {
        headers: { accept: "application/json" }, cache: "no-store", signal,
      });
      const payload = await res.json() as WordMarksPayload;
      if (guard.current.isLatest(request) && loadErrorRef.current !== null) {
        loadErrorRef.current = null;
        setLoadError(null);
      }
      if (guard.current.accept(request, payload, focus)) setSnapshot({ payload, focus });
    } catch (err) {
      // Drop failures from superseded reads too.
      if (guard.current.isLatest(request)) {
        const message = err instanceof ApiError ? err.detail : "Could not load clean-up marks.";
        loadErrorRef.current = message;
        setLoadError(message);
        onError(message);
      }
    }
  }, [runId, onError]);
  const adopt = useCallback((payload: WordMarksPayload, focus: string | null, request = guard.current.begin()) => {
    if (guard.current.accept(request, payload, focus)) setSnapshot({ payload, focus });
    if (writePending.current) {
      // The write is authoritative at its own focus. Returning to that focus
      // during the POST needs no extra read; a different focus needs one.
      refreshPending.current = !guard.current.isLatest(request) || desiredFocus.current !== focus;
    } else if (!guard.current.isLatest(request) || desiredFocus.current !== focus) void refresh();
  }, [refresh]);
  const requestFocus = useCallback((focus: string | null) => {
    if (desiredFocus.current !== focus) void refresh(focus);
  }, [refresh]);
  const beginWrite = useCallback(() => {
    writePending.current = true;
    return guard.current.begin();
  }, []);
  const finishWrite = useCallback(() => {
    writePending.current = false;
    if (refreshPending.current) {
      refreshPending.current = false;
      void refresh();
    }
  }, [refresh]);
  useEffect(() => {
    const requests = guard.current;
    void refresh(null);
    return () => requests.invalidate();
  }, [refresh]);
  return { snapshot, refresh, requestFocus, adopt, beginWrite, finishWrite, loadError };
}
