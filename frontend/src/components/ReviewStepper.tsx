import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import {
  type Segment,
  TranscriptPlayer,
  type TranscriptPlayerHandle,
} from "./TranscriptPlayer";

export interface ReviewStepperProps {
  runId: string;
  mediaUrl: string;
  segments: Segment[];
  capability: PlaybackCapability;
  lowConfidenceThreshold: number;
  // The live claim token, carried from the workbench via ?token=. null when this
  // tab does NOT hold the claim: the surface degrades to the read-only player and
  // a prompt to claim, never a broken verify button (honest UX).
  reviewToken: string | null;
  // The run's N-of-M counter at page load (verified, total-segments). Kept in
  // sync from each write's JSON response — never recomputed on the client.
  initialProgress: { verified: number; total: number };
  // Waveform strip (issue #57): forwarded to the player untouched.
  peaksUrl?: string | null;
  turns?: Turn[];
}

// A segment is a REVIEW TARGET when it has a write id and is not yet verified.
// Low-confidence segments carry the "uncertain" chip in the list (they draw the
// eye), but the terminating condition of the loop is verified — the counter is
// verified/total, so the queue empties exactly when every segment is confirmed.
function isTarget(seg: Segment): boolean {
  return seg.segmentId !== null && !seg.verified;
}

// Next review target at or after `from` (document order), else the first target
// before it (wrap once), else -1 when the queue is empty.
function nextTarget(segments: Segment[], from: number): number {
  for (let i = Math.max(from, 0); i < segments.length; i += 1) {
    if (isTarget(segments[i])) return i;
  }
  for (let i = 0; i < Math.min(from, segments.length); i += 1) {
    if (isTarget(segments[i])) return i;
  }
  return -1;
}

// Verify-and-advance triage loop (issue #53) + inline text correction (issue
// #58). Composes the pure TranscriptPlayer (playback/highlight/auto-scroll) and
// drives it through `playSegment`; owns the flag queue, the typing-guarded
// keymap, the verify/correct POSTs, the N-of-M readout, and the edit textarea.
export function ReviewStepper({
  runId,
  mediaUrl,
  segments: initialSegments,
  capability,
  lowConfidenceThreshold,
  reviewToken,
  initialProgress,
  peaksUrl,
  turns,
}: ReviewStepperProps) {
  // Own the segments so a correction re-renders its line without reaching into
  // the (pure) player. Verify/correct responses patch this array in place.
  const [segments, setSegments] = useState<Segment[]>(initialSegments);
  const [cursor, setCursor] = useState<number>(() =>
    Math.max(nextTarget(initialSegments, 0), 0),
  );
  const [progress, setProgress] = useState(initialProgress);
  const [editText, setEditText] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  // Guards re-entry synchronously: a state `busy` flips a render too late to stop
  // a second key that fires before React re-renders, so two writes could overlap
  // and apply out of order. The ref is set/cleared around the WHOLE operation
  // (fetch + parse + apply), so a second action is refused until the first has
  // fully landed. `busy` state still drives the disabled buttons.
  const busyRef = useRef<boolean>(false);
  // Set when a write 409s (claim expired or reclaimed elsewhere). The loop stops
  // and keeps the operator's place + edit — never advances against a dead claim.
  const [claimLost, setClaimLost] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // Armed when `v` is pressed with an unsaved edit in the box: the first press
  // warns (rather than silently discarding the typed text and verifying the old
  // wording); a second `v` verifies anyway. Cleared on edit, save, or move.
  const [confirmDiscard, setConfirmDiscard] = useState<boolean>(false);

  const playerRef = useRef<TranscriptPlayerHandle>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);

  const current =
    cursor >= 0 && cursor < segments.length ? segments[cursor] : null;
  const remaining = useMemo(() => segments.filter(isTarget).length, [segments]);
  const writable = reviewToken !== null && !claimLost;

  // Keep the edit box in step with the segment under the cursor (its effective
  // text — corrected-or-pipeline, already what the page rendered). Moving to a
  // fresh segment also disarms any pending discard warning.
  useEffect(() => {
    setEditText(current?.text ?? "");
    setConfirmDiscard(false);
  }, [current?.segmentId, current?.text]);

  const play = useCallback((index: number) => {
    playerRef.current?.playSegment(index);
  }, []);

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

  // Apply a write's JSON response to local state: patch the segment, refresh the
  // counter. Returns the patched array so the caller can pick the next target
  // from post-write truth (a just-verified segment is no longer a target).
  const applyResult = useCallback(
    (
      index: number,
      result: {
        verified: boolean;
        corrected: boolean;
        text: string;
        progress: { verified: number; total: number };
      },
    ): Segment[] => {
      const patched = segments.map((seg, i) =>
        i === index
          ? {
              ...seg,
              verified: result.verified,
              corrected: result.corrected,
              text: result.text,
            }
          : seg,
      );
      setSegments(patched);
      setProgress(result.progress);
      return patched;
    },
    [segments],
  );

  // POST a form-encoded write and return the parsed JSON state, or null on any
  // failure. A 409 is the stale/reclaimed claim → stop the loop (keep place +
  // edit); every other error (network, a non-JSON body from a proxy) surfaces an
  // inline message. Parsing lives INSIDE the try so a bad body is handled, never
  // an unhandled rejection. Busy is owned by the callers (they hold the ref
  // across the whole operation), so this helper does not touch it.
  const postJson = useCallback(
    async (
      path: string,
      body: Record<string, string>,
    ): Promise<Parameters<typeof applyResult>[1] | null> => {
      if (!writable || reviewToken === null) return null;
      try {
        const res = await apiFetch(path, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            // Ask for JSON explicitly: the write routes redirect a plain HTML
            // form navigation (JS-off fallback) but must return JSON to us.
            accept: "application/json",
          },
          body: new URLSearchParams({ token: reviewToken, ...body }).toString(),
        });
        return (await res.json()) as Parameters<typeof applyResult>[1];
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          // Stale/reclaimed: stop the loop, keep place + edit. The operator must
          // re-claim from the workbench; we say so rather than silently failing.
          setClaimLost(true);
        } else {
          setError(err instanceof ApiError ? err.detail : "Request failed.");
        }
        return null;
      }
    },
    [writable, reviewToken],
  );

  const verifyAndAdvance = useCallback(async () => {
    if (current?.segmentId == null || busyRef.current) return;
    // A pending edit in the box would be silently discarded by verifying the old
    // wording. Warn on the first `v`; a second one verifies anyway (explicit).
    if (editText !== (current.text ?? "") && !confirmDiscard) {
      setConfirmDiscard(true);
      return;
    }
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const index = cursor;
      const result = await postJson(
        `/review/${runId}/segments/${current.segmentId}/verify`,
        { verified: "true" },
      );
      if (!result) return;
      const patched = applyResult(index, result);
      setConfirmDiscard(false);
      const next = nextTarget(patched, index + 1);
      if (next >= 0) goTo(next);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [
    current,
    cursor,
    editText,
    confirmDiscard,
    postJson,
    runId,
    applyResult,
    goTo,
  ]);

  const saveEdit = useCallback(async () => {
    if (current?.segmentId == null || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const index = cursor;
      const result = await postJson(
        `/review/${runId}/segments/${current.segmentId}/text`,
        { text: editText },
      );
      if (!result) return;
      applyResult(index, result);
      setConfirmDiscard(false);
      // Stay on the segment after an edit (the operator likely wants to verify it
      // next); editing cleared its verified mark server-side, so it is a target
      // again and the counter reflects that.
      editRef.current?.blur();
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [current, cursor, postJson, runId, editText, applyResult]);

  // Global keymap — typing-guarded. No firing when focus is in an input/textarea
  // or with modifiers; Space and the scroll arrows are deliberately left to the
  // native <audio> and the player's own scroll handling (never rebound here).
  useEffect(() => {
    if (!writable) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const el = event.target as HTMLElement | null;
      const tag = el?.tagName;
      // Never steal a key from a form control the operator is using — the
      // textarea, but also the player's playback-speed <select> (a focused
      // <select> must not let `v` fire a verify write).
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el?.isContentEditable
      )
        return;
      switch (event.key) {
        case "v":
          event.preventDefault();
          void verifyAndAdvance();
          break;
        case "n":
          event.preventDefault();
          jumpNext();
          break;
        case "p":
          event.preventDefault();
          if (cursor >= 0) play(cursor);
          break;
        case "e":
          event.preventDefault();
          editRef.current?.focus();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [writable, verifyAndAdvance, jumpNext, play, cursor]);

  const done = remaining === 0;

  return (
    <div>
      {reviewToken === null && (
        <p className="muted" role="status">
          Not claimed by this tab — verifying and editing are disabled.{" "}
          <a href={`/review/${runId}`}>Claim this run in the workbench</a> to
          review.
        </p>
      )}
      {claimLost && (
        <p role="alert" className="tp-uncertain-chip">
          Your claim expired or was taken over. Everything you already saved is
          safe — copy any unsaved edit from the box first, then{" "}
          <a href={`/review/${runId}`}>re-claim in the workbench</a> and reopen
          this page to continue.
        </p>
      )}
      {writable && (
        <div className="review-stepper my-2" aria-label="Verify and advance">
          <p aria-live="polite">
            <strong>{progress.verified}</strong> of{" "}
            <strong>{progress.total}</strong> segments verified
            {done ? " — all done" : ` · ${remaining} left`}
          </p>
          {current && current.segmentId !== null && (
            <div>
              <p className="muted text-sm">
                Reviewing segment at {current.start.toFixed(2)}s
                {current.confidence != null &&
                  current.confidence < lowConfidenceThreshold && (
                    <span className="tp-uncertain-chip ml-2">uncertain</span>
                  )}
                {current.verified && (
                  <span className="spk-badge ml-2">verified</span>
                )}
                {current.corrected && (
                  <span className="spk-badge ml-2">edited</span>
                )}
              </p>
              <textarea
                ref={editRef}
                value={editText}
                onChange={(e) => {
                  setEditText(e.target.value);
                  setConfirmDiscard(false);
                }}
                onKeyDown={(e) => {
                  // Ctrl/Cmd+Enter saves from within the box; Escape returns keys
                  // to the loop. Plain Enter stays a newline (multi-line edits).
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                    e.preventDefault();
                    void saveEdit();
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    editRef.current?.blur();
                  }
                }}
                rows={2}
                className="w-full text-sm"
                aria-label="Corrected transcript text for this segment"
              />
              <div className="flex items-center my-1">
                <button
                  type="button"
                  onClick={() => void verifyAndAdvance()}
                  disabled={busy}
                  className="mr-2"
                >
                  Verify &amp; next <kbd>v</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => void saveEdit()}
                  disabled={busy}
                  className="mr-2"
                >
                  Save edit <kbd>Ctrl/⌘+↵</kbd>
                </button>
                <button
                  type="button"
                  onClick={jumpNext}
                  disabled={busy}
                  className="mr-2"
                >
                  Skip <kbd>n</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => cursor >= 0 && play(cursor)}
                  disabled={busy}
                >
                  Replay <kbd>p</kbd>
                </button>
              </div>
              {confirmDiscard && (
                <p role="alert" className="text-sm">
                  You have an unsaved edit. Press <kbd>Ctrl/⌘+↵</kbd> to save
                  it, or <kbd>v</kbd> again to verify the original wording and
                  discard the edit.
                </p>
              )}
              {error && (
                <p role="alert" className="text-sm">
                  {error}
                </p>
              )}
            </div>
          )}
        </div>
      )}
      <TranscriptPlayer
        ref={playerRef}
        runId={runId}
        mediaUrl={mediaUrl}
        segments={segments}
        capability={capability}
        lowConfidenceThreshold={lowConfidenceThreshold}
        // Clicking a transcript line moves the edit cursor there (and plays it),
        // so a correction always lands on the segment the operator is looking at
        // — and a segment already verified can be reached again to fix. Only when
        // this tab holds the claim; the read-only surface passes no callback.
        onSegmentSelect={writable ? setCursor : undefined}
        // Waveform strip (issue #57): the review surface additionally shows the
        // cursor marker so region clicks and the keymap stay visibly in sync.
        peaksUrl={peaksUrl}
        turns={turns}
        cursorIndex={writable ? cursor : undefined}
      />
    </div>
  );
}
