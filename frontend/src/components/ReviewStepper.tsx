import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import { KeymapHelp } from "./KeymapHelp";
import {
  type Segment,
  type SplitWord,
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
  // The assignable speaker roster for the per-child reassign picker (issue #59
  // slice 3). ACTIVE identities only (the server filters merged/archived out, so
  // the picker never offers a speaker the /relabel write would reject). Empty on
  // a run with no roster yet — the picker then offers only "inherit / reset".
  speakers: { id: string; displayName: string }[];
}

// A fresh idempotency nonce for a /relabel write (issue #59 slice 3). Uses
// crypto.getRandomValues — NOT crypto.randomUUID — deliberately: this is a
// self-hosted tool an operator may reach over plain HTTP on a LAN, where
// randomUUID (secure-context only) is undefined, while getRandomValues works
// everywhere. 16 bytes → a 32-char hex string, inside the route's [8, 64] bound.
function makeNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// A segment is a REVIEW TARGET when it is the queue entry for its parent and is
// not yet verified. `reviewTarget` is true on exactly one line per parent (issue
// #59): an unsplit line, or the FIRST child of a split parent — so a split
// parent's children never double-count the queue. Low-confidence segments carry
// the "uncertain" chip in the list (they draw the eye), but the terminating
// condition of the loop is verified — the counter is verified/total, so the
// queue empties exactly when every parent is confirmed.
function isTarget(seg: Segment): boolean {
  return seg.reviewTarget && !seg.verified;
}

// How many rendered lines share this parent id (issue #59). >1 ⇒ the parent has
// been split into derived children; splitting further and editing are disabled
// on such a parent this slice, and its children keep their word-derived text.
function siblingCount(
  segments: Segment[],
  sourceSegmentId: string | null,
): number {
  if (sourceSegmentId === null) return 0;
  return segments.filter((s) => s.sourceSegmentId === sourceSegmentId).length;
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
  speakers,
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
  // Split mode (issue #59): an explicit toggle — word clicks split a segment only
  // while ON, so ordinary line/word/waveform clicks stay non-destructive (they
  // select/seek). `splitData` is the focused segment's lazily-fetched words +
  // splittability, refreshed whenever the focus or mode changes. Null while off,
  // between fetches, or for an already-split parent (no further split this slice).
  const [splitMode, setSplitMode] = useState<boolean>(false);
  const [splitData, setSplitData] = useState<{
    segmentId: string;
    splittable: boolean;
    reason: string | null;
    words: SplitWord[];
  } | null>(null);
  // The `?` cheat-sheet overlay (issue #51). Opened by the `?` key or the visible
  // "⌨ Shortcuts" button; closed by Escape, its close button, or a backdrop click.
  const [helpOpen, setHelpOpen] = useState<boolean>(false);

  const playerRef = useRef<TranscriptPlayerHandle>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);

  const current =
    cursor >= 0 && cursor < segments.length ? segments[cursor] : null;
  const remaining = useMemo(() => segments.filter(isTarget).length, [segments]);
  const writable = reviewToken !== null && !claimLost;
  // The focused line's parent (its write target) and whether that parent has
  // already been split — the latter gates both re-splitting and editing (issue
  // #59): a split parent's text is word-derived, so free-form correction is
  // mutually exclusive with it (the /text route 409s this; we never show a
  // button that will fail).
  const focusParentId = current?.sourceSegmentId ?? null;
  const isSplitParent = siblingCount(segments, focusParentId) > 1;

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

  // Apply a verify/correct JSON response to local state: patch the segment,
  // refresh the counter. Returns the patched array so the caller can pick the
  // next target from post-write truth (a just-verified segment is no longer a
  // target).
  //
  // The write targets a PARENT id, so the response's verified/corrected flags
  // apply to EVERY line sharing that parent — a split parent's derived children
  // are verified together (issue #59). But the response's `text` is the parent's
  // full effective text; a split parent's children keep their own word-derived
  // text and must NOT be clobbered with it. So: patch flags across all siblings;
  // adopt the response text only in the unsplit single-line case.
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
        };
      });
      setSegments(patched);
      setProgress(result.progress);
      return patched;
    },
    [segments],
  );

  // POST a form-encoded write and return its parsed JSON body (of whatever
  // shape), or null on any failure. Parsing lives INSIDE the try so a bad body is
  // handled, never an unhandled rejection. Busy is owned by the callers (they
  // hold the ref across the whole operation), so this helper does not touch it.
  //
  // A 409 has TWO meanings on these routes and the caller says which applies. For
  // verify/correct it is always a stale/reclaimed claim (default) → stop the loop
  // and prompt a re-claim. For /split it can instead be a segment-STATE conflict
  // (e.g. the parent was corrected in another tab between the words fetch and the
  // click); `claimLostOnConflict:false` surfaces the server's honest reason inline
  // and keeps the claim, rather than falsely locking the whole surface. Every
  // other error (network, a non-JSON body from a proxy) surfaces inline.
  const postForm = useCallback(
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
            // Ask for JSON explicitly: the write routes redirect a plain HTML
            // form navigation (JS-off fallback) but must return JSON to us.
            accept: "application/json",
          },
          body: new URLSearchParams({ token: reviewToken, ...body }).toString(),
        });
        return (await res.json()) as T;
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          if (err.conflictKind === "claim") {
            // The server explicitly marked this 409 as a lost/taken claim (even on
            // a route whose OTHER 409s are state conflicts): stop the loop and
            // prompt a re-claim, regardless of claimLostOnConflict.
            setClaimLost(true);
          } else if (opts?.claimLostOnConflict === false) {
            // Segment-state conflict, not a claim loss: show the server reason
            // (e.g. "cannot split a corrected segment") and keep the claim.
            setError(err.detail);
          } else {
            // Stale/reclaimed: stop the loop, keep place + edit. The operator must
            // re-claim from the workbench; we say so rather than silently failing.
            setClaimLost(true);
          }
        } else {
          setError(err instanceof ApiError ? err.detail : "Request failed.");
        }
        return null;
      }
    },
    [writable, reviewToken],
  );

  // Verify/correct writes: the parent-scoped review-state shape applyResult eats.
  const postJson = useCallback(
    (
      path: string,
      body: Record<string, string>,
    ): Promise<Parameters<typeof applyResult>[1] | null> =>
      postForm<Parameters<typeof applyResult>[1]>(path, body),
    [postForm],
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
    // Never POST /text for a split parent — it would 409 (split and correction
    // are mutually exclusive). The Save button is already disabled for this case;
    // this guards the keyboard path (Ctrl/⌘+↵) too.
    if (siblingCount(segments, current.sourceSegmentId) > 1) return;
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
  }, [current, cursor, postJson, runId, editText, applyResult, segments]);

  // Lazily fetch the focused segment's words when split mode is engaged (issue
  // #59) — the /words payload is never folded into the shared hydration props, so
  // it costs nothing until the operator actually splits. Skipped for an already-
  // split parent (no further split this slice) and re-run on every focus change.
  // The abort + parent-id guard keep a superseded response from painting a line.
  // `current?.corrected` is a dep so that correcting the focused segment (which
  // makes it unsplittable server-side) forces a refetch — otherwise the cut
  // points would stay live and a click would 409 on the now-corrected segment.
  useEffect(() => {
    if (!splitMode || !writable || focusParentId === null || isSplitParent) {
      setSplitData(null);
      return;
    }
    const parentId = focusParentId;
    // Clear first so the status line reads "Loading words…" during a refetch,
    // never the previous segment's stale splittability (honest UX). The cut
    // buttons are already parent-id guarded, but the status line is not.
    setSplitData(null);
    const controller = new AbortController();
    void (async () => {
      try {
        const res = await apiFetch(
          `/review/${runId}/segments/${parentId}/words`,
          {
            headers: { accept: "application/json" },
            signal: controller.signal,
          },
        );
        const data = (await res.json()) as {
          splittable: boolean;
          reason: string | null;
          words: SplitWord[];
        };
        if (!controller.signal.aborted) {
          // Pick fields explicitly — the payload also carries its own segmentId,
          // and a blind spread would overwrite the parent-id guard value.
          setSplitData({
            segmentId: parentId,
            splittable: data.splittable,
            reason: data.reason,
            words: data.words,
          });
        }
      } catch (err) {
        // An abort is the effect superseding itself — not an error to surface.
        if (controller.signal.aborted || (err as Error).name === "AbortError") {
          return;
        }
        setSplitData({
          segmentId: parentId,
          splittable: false,
          reason: "Could not load words for splitting.",
          words: [],
        });
      }
    })();
    return () => {
      controller.abort();
    };
  }, [
    splitMode,
    writable,
    focusParentId,
    isSplitParent,
    runId,
    current?.corrected,
  ]);

  // Split the focused parent BEFORE word `wordIndex` (issue #59). The response is
  // a whole-run reconcile (segments + progress, the hydration shape), so we adopt
  // it wholesale and re-seat the cursor on the parent's queue entry (its first
  // child) via setCursor — WITHOUT play(): a split is a structural edit, not a
  // navigation, so we don't restart audio the way verifyAndAdvance's goTo does.
  // busyRef mirrors the verify/correct guard so a double word-click can't race
  // two splits. A structurally-idempotent backend makes a replay a no-op.
  const splitAt = useCallback(
    async (sourceSegmentId: string, wordIndex: number) => {
      if (busyRef.current) return;
      // A pending edit in the box would be silently discarded by the split's
      // whole-run reconcile (it re-syncs the edit box to the child's word-derived
      // text). Warn on the first click; a second word click splits anyway — the
      // same explicit-discard contract verifyAndAdvance uses for `v`.
      if (current && editText !== (current.text ?? "") && !confirmDiscard) {
        setConfirmDiscard(true);
        return;
      }
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const result = await postForm<{
          segments: Segment[];
          progress: { verified: number; total: number };
        }>(
          `/review/${runId}/segments/${sourceSegmentId}/split`,
          { word_index: String(wordIndex) },
          // A 409 here is a segment-STATE conflict (the parent was corrected
          // out from under us), not a claim loss: show the reason, keep the
          // claim, do not lock the surface.
          { claimLostOnConflict: false },
        );
        if (!result) return;
        setConfirmDiscard(false);
        setSegments(result.segments);
        setProgress(result.progress);
        const targetIdx = result.segments.findIndex(
          (s) => s.sourceSegmentId === sourceSegmentId && s.reviewTarget,
        );
        setCursor(
          targetIdx >= 0
            ? targetIdx
            : Math.max(nextTarget(result.segments, 0), 0),
        );
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [postForm, runId, current, editText, confirmDiscard],
  );

  // Reassign ONE derived split child to a roster speaker — or reset it to inherit
  // its label (issue #59 slice 3). Posts the child's word range to /relabel with a
  // fresh idempotency nonce. Like splitAt, the response is a whole-run reconcile (a
  // child's speaker string moved, which the per-segment shape can't carry), so we
  // adopt segments + progress wholesale WITHOUT moving the cursor or restarting
  // audio — reassignment is an attribution edit, not a navigation. busyRef mirrors
  // the verify/split guard so a second pick can't race two writes. A 409 here is a
  // segment-STATE conflict (the range is no longer a current child — a re-split
  // landed elsewhere), NOT a claim loss: show the server's reason and keep the
  // claim, rather than falsely locking the surface.
  const reassignChild = useCallback(
    async (seg: Segment, speakerId: string | null) => {
      if (busyRef.current) return;
      if (
        seg.sourceSegmentId === null ||
        seg.wordStart === null ||
        seg.wordEnd === null
      )
        return;
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const body: Record<string, string> = {
          nonce: makeNonce(),
          action: speakerId === null ? "inherit" : "assign",
          start_word_index: String(seg.wordStart),
          end_word_index: String(seg.wordEnd),
        };
        if (speakerId !== null) body.speaker_id = speakerId;
        const result = await postForm<{
          segments: Segment[];
          progress: { verified: number; total: number };
        }>(
          `/review/${runId}/segments/${seg.sourceSegmentId}/relabel`,
          body,
          { claimLostOnConflict: false },
        );
        if (!result) return;
        setSegments(result.segments);
        setProgress(result.progress);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [postForm, runId],
  );

  // Assign the WHOLE focused segment to a roster speaker — or reset it to inherit
  // its label (issue #51): the clickable/keyboard twin of the per-child reassign
  // above. Posts to /relabel WITHOUT word indices, so the two-scope route
  // (app.py) rules on the whole segment. Refused on a split parent — its children
  // carry their own word-scoped rulings and a whole-segment ruling must not
  // attract new attribution there (the same reason its edit/re-split are gated).
  // Like reassignChild the response is a whole-run reconcile adopted WITHOUT
  // moving the cursor or restarting audio — attribution, not navigation — and a
  // 409 is a segment-STATE conflict (keep the claim), not a claim loss.
  const reassignSegment = useCallback(
    async (speakerId: string | null) => {
      if (busyRef.current) return;
      if (focusParentId === null || isSplitParent) return;
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const body: Record<string, string> = {
          nonce: makeNonce(),
          action: speakerId === null ? "inherit" : "assign",
        };
        if (speakerId !== null) body.speaker_id = speakerId;
        const result = await postForm<{
          segments: Segment[];
          progress: { verified: number; total: number };
        }>(
          `/review/${runId}/segments/${focusParentId}/relabel`,
          body,
          { claimLostOnConflict: false },
        );
        if (!result) return;
        setSegments(result.segments);
        setProgress(result.progress);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [postForm, runId, focusParentId, isSplitParent],
  );

  // Global keymap — typing-guarded. No firing when focus is in an input/textarea
  // or with modifiers; Space and the scroll arrows are deliberately left to the
  // native <audio> and the player's own scroll handling (never rebound here).
  // Unmodified keys only (Shift is allowed, so `?` and the digits pass): v/n/p/e
  // drive verify/skip/replay/edit (issue #53); j/k walk to the next/previous
  // segment; 1–9 assign the focused segment to the Nth roster speaker and 0 resets
  // it to inherit; `?` opens the cheat-sheet (issue #51). Every one of these has a
  // clickable equivalent on the page.
  useEffect(() => {
    if (!writable) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      // While the cheat-sheet dialog is open its close button holds focus — a
      // <button>, which the typing-guard below does NOT block — so suppress the
      // global loop here, or `v`/digits would fire behind the modal. The dialog
      // owns its own Escape/Tab handling.
      if (helpOpen) return;
      const el = event.target as HTMLElement | null;
      const tag = el?.tagName;
      // Never steal a key from a form control the operator is using — the
      // textarea, but also the player's playback-speed <select> and the speaker
      // pickers (a focused <select> must not let `v` fire a verify write).
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
        case "j":
          // Next segment (plays on move, like clicking a line). Clamped so the
          // last segment stays put rather than blanking the cursor.
          event.preventDefault();
          goTo(Math.min(cursor + 1, segments.length - 1));
          break;
        case "k":
          // Previous segment — the "go back" the forward-only n/jumpNext lacks.
          event.preventDefault();
          goTo(Math.max(cursor - 1, 0));
          break;
        case "0":
          // Reset the focused segment to inherit its label (no-op on a split
          // parent or a roster-less run; reassignSegment guards both).
          event.preventDefault();
          void reassignSegment(null);
          break;
        case "?":
          event.preventDefault();
          setHelpOpen(true);
          break;
        default:
          // Digit-assign (issue #51): 1–9 → the Nth roster speaker. Only fires on
          // a real roster slot, so a run with fewer speakers never writes a
          // phantom ruling; reassignSegment additionally refuses a split parent.
          if (event.key >= "1" && event.key <= "9") {
            const speaker = speakers[Number(event.key) - 1];
            if (speaker) {
              event.preventDefault();
              void reassignSegment(speaker.id);
            }
          }
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [
    writable,
    helpOpen,
    verifyAndAdvance,
    jumpNext,
    play,
    goTo,
    cursor,
    segments,
    reassignSegment,
    speakers,
  ]);

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
          <button
            type="button"
            onClick={() => setSplitMode((on) => !on)}
            aria-pressed={splitMode}
            className="text-sm my-1"
          >
            {splitMode ? "Exit split mode" : "⎇ Split at a word"}
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            aria-haspopup="dialog"
            className="text-sm my-1 ml-2"
          >
            ⌨ Shortcuts
          </button>
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
                disabled={isSplitParent}
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
                className="w-full text-sm disabled:opacity-60"
                aria-label="Corrected transcript text for this segment"
              />
              {isSplitParent && (
                <p className="muted text-sm" role="note">
                  This segment is split, so editing is disabled — splitting and
                  free-form correction are mutually exclusive. (Re-transcribe to
                  clear the split.)
                </p>
              )}
              {splitMode && !isSplitParent && (
                <p className="text-sm" role="status">
                  {splitData == null
                    ? "Loading words…"
                    : splitData.splittable
                      ? "Split mode on — click a word to cut the segment before it."
                      : (splitData.reason ??
                        "This segment can’t be split at a word boundary.")}
                </p>
              )}
              {splitMode && isSplitParent && (
                <p className="muted text-sm" role="status">
                  This segment is already split — it can’t be split further in
                  this release.
                </p>
              )}
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
                  disabled={busy || isSplitParent}
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
              {!isSplitParent && (
                // The clickable twin of the 1–9 / 0 digit-assign keys (issue #51):
                // an action <select> that assigns the WHOLE segment to a roster
                // speaker, or resets it to its detected label. Value is pinned to
                // "" so it always returns to the placeholder after a pick (it fires
                // an action, it does not reflect the segment's current speaker).
                // Hidden on a split parent, whose children carry their own scoped
                // pickers. The keymap's SELECT guard keeps digits from firing while
                // this control is focused.
                <label className="tp-reassign text-sm my-1 block">
                  Assign speaker:{" "}
                  <select
                    className="text-sm"
                    value=""
                    disabled={busy}
                    aria-label="Assign a speaker to this whole segment"
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val === "") return;
                      void reassignSegment(val === "__inherit__" ? null : val);
                    }}
                  >
                    <option value="">Assign speaker…</option>
                    {speakers.map((sp, i) => (
                      <option key={sp.id} value={sp.id}>
                        {i < 9 ? `${i + 1}. ` : ""}
                        {sp.displayName}
                      </option>
                    ))}
                    <option value="__inherit__">Reset to detected speaker</option>
                  </select>
                </label>
              )}
              {confirmDiscard && (
                <p role="alert" className="text-sm">
                  You have an unsaved edit. Press <kbd>Ctrl/⌘+↵</kbd> to save
                  it, or repeat the action (<kbd>v</kbd> to verify, or click the
                  word again to split) to discard the edit and continue.
                </p>
              )}
              {error && (
                <p role="alert" className="text-sm">
                  {error}
                </p>
              )}
            </div>
          )}
          <KeymapHelp
            open={helpOpen}
            onClose={() => setHelpOpen(false)}
            hasRoster={speakers.length > 0}
          />
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
        // Split mode (issue #59): render the focused line's words as clickable cut
        // points only when split mode is on, the segment is splittable, and the
        // fetched words belong to the segment now under the cursor (a stale fetch
        // never paints the wrong line). onSplitAt raises the chosen cut here.
        splitFocus={
          splitMode &&
          !isSplitParent &&
          cursor >= 0 &&
          splitData !== null &&
          splitData.splittable &&
          splitData.segmentId === focusParentId
            ? {
                segmentIndex: cursor,
                sourceSegmentId: splitData.segmentId,
                words: splitData.words,
              }
            : null
        }
        onSplitAt={writable ? (id, wi) => void splitAt(id, wi) : undefined}
        // Per-child reassign (issue #59 slice 3): a split parent's derived child
        // lines (those carrying wordStart/wordEnd) render a speaker <select>.
        // Only when this tab holds the claim; the read-only surface passes none,
        // so it stays byte-identical (no picker). busy disables the picker mid-
        // write, sharing the verify/split/save guard.
        reassignSpeakers={writable ? speakers : undefined}
        onReassign={
          writable ? (seg, speakerId) => void reassignChild(seg, speakerId) : undefined
        }
        reassignBusy={busy}
      />
    </div>
  );
}
