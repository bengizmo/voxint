import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  FALLBACK_ANNOTATION_LIMITS,
  type AnnotationLimits,
  type AnnotationShape,
  type AnnotationTagShape,
} from "../lib/annotations";
import { ApiError, apiFetch } from "../lib/api-client";
import { writeClipboard } from "../lib/clipboard";
import { makeNonce } from "../lib/nonce";
import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import { useAnnotations } from "./AnnotationLayer";
import { KeymapHelp } from "./KeymapHelp";
import {
  ASSIGN_DIGIT_MAX,
  ASSIGN_DIGIT_MIN,
  isSaveEditChord,
  REVIEW_KEY,
  SAVE_EDIT_LABEL,
} from "./keymap";
import {
  type ReconciliationEntry,
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
  // Run-level declared-rule reconciliation (issue #83): one row per rule the run's
  // frozen domain pack DECLARED, with whether it materially fired anywhere. Feeds
  // the "declared but never fired" panel, which renders only when this is non-empty
  // (a run with no pack, or a pack with no corrections, sends []). Computed once
  // server-side over the immutable raw_text — never recomputed on the client.
  reconciliation: ReconciliationEntry[];
  // Operator annotation layer (issue #86). The run's existing annotations + the
  // global tag list (both the byte-identical GET /review/{id}/annotations shape),
  // the server-enforced caps the toolbar mirrors, and the per-action CSRF token for
  // creating a global tag. Reads hydrate for every viewer; creation is claim-gated
  // client-side. Defaulted so a props payload without them simply shows no layer.
  annotations?: AnnotationShape[];
  annotationTags?: AnnotationTagShape[];
  annotationLimits?: AnnotationLimits;
  tagCsrf?: string | null;
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
  // Default to empty: only the review route computes reconciliation, and a defensive
  // default keeps the panel simply absent (never a crash) if a props payload omits it.
  reconciliation = [],
  // Annotation layer (issue #86): defaulted so an older props payload just shows an
  // empty layer rather than crashing. The review route always sends real values.
  annotations: initialAnnotations = [],
  annotationTags: initialAnnotationTags = [],
  annotationLimits = FALLBACK_ANNOTATION_LIMITS,
  tagCsrf = null,
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
  // A synchronous mirror of helpOpen for the global keydown handler. State flips a
  // render too late (the same reason busyRef exists): between opening the dialog
  // and the effect re-subscribing, the still-registered listener would see the old
  // `false` and let `v`/a digit fire behind the just-opened modal. The ref, written
  // during render, is always current when a key arrives.
  const helpOpenRef = useRef<boolean>(false);
  helpOpenRef.current = helpOpen;
  // A polite status line for the whole-segment speaker actions (issue #51): a
  // fire-and-reset control and the digit keys otherwise change only the inline
  // speaker name — not in any live region — so a keyboard/screen-reader operator
  // gets no confirmation. Errors already speak (role="alert"); this makes success
  // speak too. Cleared as soon as the operator navigates or edits.
  const [assignStatus, setAssignStatus] = useState<string | null>(null);
  // Issue #83 disclosure state. `provOpen` expands the current segment's domain-pack
  // rule list; `rawOpen` reveals the raw-vs-corrected compare beside the edit box.
  // Both reset on segment change (in the edit-sync effect). `copyStatus` is a polite,
  // transient line for the copy-raw affordance (success or clipboard-unavailable
  // fallback). `reconOpen` toggles the run-level "declared but never fired" panel.
  const [provOpen, setProvOpen] = useState<boolean>(false);
  const [rawOpen, setRawOpen] = useState<boolean>(false);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const [reconOpen, setReconOpen] = useState<boolean>(false);

  const playerRef = useRef<TranscriptPlayerHandle>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);
  // The annotation layer resolves a text selection relative to this root (an
  // ancestor of the transcript lines); attached to the outer div below.
  const annotationRootRef = useRef<HTMLDivElement>(null);
  // A bridge to useAnnotations().reload — the hook is created AFTER the transcript
  // write callbacks that must trigger it (text edit / split / relabel change the
  // render, so annotations must re-resolve). The ref decouples that ordering.
  const reloadAnnotationsRef = useRef<(() => Promise<void>) | null>(null);

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
    // A speaker-assignment announcement belongs to the segment it was made on;
    // drop it as soon as the operator moves so it never trails onto another line.
    setAssignStatus(null);
    // The #83 per-segment disclosures (provenance rule list, raw compare, copy
    // status) belong to the focused segment — collapse/clear them on any move so
    // one segment's raw text or rule list never lingers over another.
    setProvOpen(false);
    setRawOpen(false);
    setCopyStatus(null);
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
          // Operator-edit supersedes deterministic provenance (issue #83): once the
          // operator saves their own text via /text, the domain-pack correction
          // trace's spans no longer address the effective text, so the "corrected by
          // domain pack" marker would be stale (and misleading — it is a PIPELINE
          // edit, not this operator's). Clear it on a text-save — but ONLY when the
          // save actually left a correction (`result.corrected`): a save that reverts
          // to the pipeline text clears the correction server-side (corrected=false),
          // where the pack provenance is valid again and the server keeps emitting it,
          // so mirror that here rather than hiding it until reload. A plain verify
          // (supersedeProvenance unset) always leaves the provenance intact.
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
      applyResult(index, result, { supersedeProvenance: true });
      setConfirmDiscard(false);
      // The edited text moved: re-resolve annotations so their marks repaint (or
      // drop to a stale locator) against the new wording instead of clinging to
      // pre-edit offsets until a page reload (issue #86).
      void reloadAnnotationsRef.current?.();
      // Stay on the segment after an edit (the operator likely wants to verify it
      // next); editing cleared its verified mark server-side, so it is a target
      // again and the counter reflects that.
      editRef.current?.blur();
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [current, cursor, postJson, runId, editText, applyResult, segments]);

  // Copy the current segment's immutable raw text to the clipboard (issue #83).
  // navigator.clipboard is undefined on a non-secure LAN context (plain http) —
  // exactly where a self-hosted operator often runs — and writeText can also reject,
  // so BOTH paths fall back to an honest instruction to select the shown raw text
  // manually (it lives in a readOnly, selectable region). Never claims success it
  // did not achieve.
  const copyRaw = useCallback(async () => {
    const raw = current?.rawText;
    if (raw == null) return;
    setRawOpen(true);
    if (await writeClipboard(raw)) {
      setCopyStatus("Raw text copied to the clipboard.");
    } else {
      setCopyStatus(
        "Couldn’t copy automatically — select the raw text above and copy it manually.",
      );
    }
  }, [current?.rawText]);

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
        // The split rewrote the rendered lines: re-resolve annotations so their
        // marks land on the correct new line indices, not the pre-split ones.
        void reloadAnnotationsRef.current?.();
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
        // Relabel changed speaker attribution: re-resolve so the panel's per-row
        // speaker labels follow the new assignment instead of going stale.
        void reloadAnnotationsRef.current?.();
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
      if (focusParentId === null) return;
      if (isSplitParent) {
        // The 1–9 / 0 keys stay live on a split parent (its clickable <select> is
        // hidden there), so refusing silently would be a promised key that quietly
        // does nothing. Say why — each split part carries its own speaker picker.
        setError(
          "This segment is split — assign speakers on each part with its own picker.",
        );
        return;
      }
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
        // A relabel is attribution-only: it never rewrites a segment's text,
        // identity, or document order, so the edit-resync effect (keyed on
        // segmentId + text) does NOT fire and a pending textarea edit survives it.
        // That is why — unlike splitAt, whose reconcile can rewrite the focused
        // text — this path needs no confirmDiscard guard.
        setSegments(result.segments);
        setProgress(result.progress);
        // Relabel changed speaker attribution: re-resolve so the panel's per-row
        // speaker labels follow the new assignment instead of going stale.
        void reloadAnnotationsRef.current?.();
        // Announce the outcome (issue #51): the inline speaker name changes but is
        // not in a live region, so a screen-reader operator would otherwise hear
        // nothing on success while failures already speak via role="alert".
        const name = speakers.find((s) => s.id === speakerId)?.displayName;
        setAssignStatus(
          speakerId === null
            ? "Reset to the detected speaker."
            : name != null
              ? `Assigned to ${name}.`
              : "Speaker assigned.",
        );
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [postForm, runId, focusParentId, isSplitParent, speakers],
  );

  // A claim-loss during an annotation write bubbles here so the whole surface stops
  // as one, exactly as a verify/correct claim loss would.
  const onAnnotationClaimLost = useCallback(() => setClaimLost(true), []);

  // Operator annotation layer (issue #86): owns the highlights/tags state, the
  // selection toolbar, and the Highlights panel. It feeds the player its per-line
  // highlight spans + stale locators + the mouseup selection signal, and jumps
  // reuse the same goTo the review loop uses.
  const {
    spansByLine: annotationSpans,
    staleLines: annotationStaleLines,
    captureSelection: annotationCapture,
    annotateFromKeyboard: annotateHotkey,
    reload: reloadAnnotations,
    toolbar: annotationToolbar,
    panel: annotationPanel,
  } = useAnnotations({
    runId,
    reviewToken,
    writable,
    rootRef: annotationRootRef,
    initialAnnotations,
    initialTags: initialAnnotationTags,
    limits: annotationLimits,
    tagCsrf,
    onJump: goTo,
    onClaimLost: onAnnotationClaimLost,
  });
  // Keep the write callbacks' bridge pointed at the live reload (see the ref decl).
  useEffect(() => {
    reloadAnnotationsRef.current = reloadAnnotations;
  }, [reloadAnnotations]);

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
      // global loop here, or `v`/digits would fire behind the modal. Read the ref,
      // not the state closure, so a key arriving in the render-gap right after the
      // dialog opens is still suppressed. The dialog owns its own Escape/Tab.
      if (helpOpenRef.current) return;
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
      // Lower-case the letter keys so Caps Lock doesn't silently disable the
      // shortcuts; `?` and the digit branch below read the raw key unaffected.
      // Key literals come from REVIEW_KEY (shared with the cheat-sheet and the
      // inline <kbd> hints) so a rebinding can't drift; dispatch stays a plain,
      // visible switch.
      switch (event.key.toLowerCase()) {
        case REVIEW_KEY.verify:
          event.preventDefault();
          void verifyAndAdvance();
          break;
        case REVIEW_KEY.skip:
          event.preventDefault();
          jumpNext();
          break;
        case REVIEW_KEY.replay:
          event.preventDefault();
          if (cursor >= 0) play(cursor);
          break;
        case REVIEW_KEY.edit:
          event.preventDefault();
          editRef.current?.focus();
          break;
        case REVIEW_KEY.next: {
          // Next segment (plays on move, like clicking a line). Clamped, and a
          // no-op at the last segment — no needless replay of the current one.
          event.preventDefault();
          const next = Math.min(cursor + 1, segments.length - 1);
          if (next !== cursor) goTo(next);
          break;
        }
        case REVIEW_KEY.previous: {
          // Previous segment — the "go back" the forward-only n/jumpNext lacks.
          event.preventDefault();
          const prev = Math.max(cursor - 1, 0);
          if (prev !== cursor) goTo(prev);
          break;
        }
        case REVIEW_KEY.resetSpeaker:
          // Reset the focused segment to inherit its detected label. Fires
          // regardless of roster (inherit needs no speaker); reassignSegment
          // refuses only a split parent (with an inline reason).
          event.preventDefault();
          void reassignSegment(null);
          break;
        case REVIEW_KEY.help:
          event.preventDefault();
          setHelpOpen(true);
          break;
        case REVIEW_KEY.annotate:
          // Highlight the selected transcript text (issue #86). An honest no-op
          // status shows when nothing is selected — the toolbar needs a selection.
          event.preventDefault();
          annotateHotkey();
          break;
        default:
          // Digit-assign (issue #51): 1–9 → the Nth roster speaker. Only fires on
          // a real roster slot, so a run with fewer speakers never writes a
          // phantom ruling; reassignSegment additionally refuses a split parent.
          if (
            event.key >= String(ASSIGN_DIGIT_MIN) &&
            event.key <= String(ASSIGN_DIGIT_MAX)
          ) {
            const speaker = speakers[Number(event.key) - ASSIGN_DIGIT_MIN];
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
    verifyAndAdvance,
    jumpNext,
    play,
    goTo,
    cursor,
    segments,
    reassignSegment,
    speakers,
    annotateHotkey,
  ]);

  const done = remaining === 0;

  // Issue #83 derived view state for the current segment. `corrections` is the
  // pipeline provenance (null once superseded by an operator edit — see
  // applyResult); `shownEntries` are the rules that materially fired (only when the
  // trace is readable — a version mismatch yields status "unavailable" instead).
  // `appliedCount`/`neverFiredCount`/`skippedCount` split the run's declared rules
  // into the panel's three honest states: applied, `no_raw_match` ("never fired" —
  // the term wasn't in the recording), and `growth_rejected` ("skipped" — it DID
  // match raw text but the change overflowed the growth ceiling). The collapsed
  // summary must not lump the last two together, or it contradicts its own per-row
  // badges and tells the operator a matched term "never fired".
  const corrections = current?.corrections ?? null;
  const shownEntries =
    corrections?.status === "shown" ? corrections.entries : [];
  const appliedCount = reconciliation.filter(
    (r) => r.status === "applied",
  ).length;
  const neverFiredCount = reconciliation.filter(
    (r) => r.status === "no_raw_match",
  ).length;
  const skippedCount = reconciliation.filter(
    (r) => r.status === "growth_rejected",
  ).length;

  return (
    <div ref={annotationRootRef}>
      {/* The unclaimed notice is server-rendered OUTSIDE this island (the
          template owns it, JS on or off) — rendering a copy here doubled it up
          after hydration (review finding). */}
      {claimLost && (
        // A full-sentence alert, so the .notice box — never the nowrap
        // uppercase .tp-uncertain-chip, which cannot wrap and overflowed the
        // stepper card (review finding).
        <p role="alert" className="notice text-sm">
          Your claim expired or was taken over. Everything you already saved is
          safe — copy any unsaved edit from the box first, then{" "}
          <a href={`/review/${runId}`}>re-claim in the workbench</a> and reopen
          this page to continue.
        </p>
      )}
      {/* The stepper card renders for every viewer: progress is run context a
          read-only/claim-lost tab keeps (and the JS-off fallback shows) — only
          the write controls below are claim-gated (review finding). */}
      <section className="review-stepper my-2" aria-label="Verify and advance">
        {/* Progress track (issue #92): the count stays the visible text signal;
            the bar is aria-hidden decoration driven by the same numbers.
            aria-atomic so a change announces the whole sentence, never a bare
            number. */}
        <div className="progress-wrap">
          <p aria-live="polite" aria-atomic="true">
            <strong>{progress.verified}</strong> of{" "}
            <strong>{progress.total}</strong> segments verified
            {done ? " — all done" : ` · ${remaining} left`}
          </p>
          <span className="progress-track" aria-hidden="true">
            <span
              style={{
                width: `${progress.total > 0 ? (progress.verified / progress.total) * 100 : 0}%`,
              }}
            />
          </span>
        </div>
        {writable && (
          <div>
          {/* Run-level declared-rule reconciliation (issue #83): a collapsible
              summary of which of the run's domain-pack correction rules actually
              fired. Renders only when the run declared corrections (empty ⇒ no
              pack or no rules ⇒ nothing to reconcile). Read-only run context, not a
              per-segment control, so it lives in the header above the loop. */}
          {reconciliation.length > 0 && (
            <div className="review-reconciliation my-1">
              <button
                type="button"
                onClick={() => setReconOpen((on) => !on)}
                aria-expanded={reconOpen}
                aria-controls="review-reconciliation-body"
                className="text-sm"
              >
                {reconOpen ? "▾" : "▸"} Correction rules —{" "}
                {appliedCount} of {reconciliation.length} applied
                {neverFiredCount > 0 ? `, ${neverFiredCount} never fired` : ""}
                {skippedCount > 0 ? `, ${skippedCount} skipped` : ""}
              </button>
              {reconOpen && (
                <div id="review-reconciliation-body" className="text-sm my-1">
                  <p className="muted">
                    Each rule declared by this run’s domain pack, and whether it
                    matched and applied on any segment’s raw ASR text. A rule that
                    never fired usually means the recording didn’t contain the term
                    it matches — or the term was split across a pause; declare such
                    terms as <code>vocabulary</code> so they’re biased at
                    transcription time instead. A <em>skipped</em> rule did match but
                    its replacement would have over-lengthened the segment.
                  </p>
                  <ul className="review-reconciliation-list">
                    {reconciliation.map((r) => (
                      <li key={r.id}>
                        <code>{r.match}</code> → <code>{r.replace}</code>{" "}
                        <span className="spk-badge ml-1">
                          {r.status === "applied"
                            ? `applied · ${r.appliedCount} segment${r.appliedCount === 1 ? "" : "s"}`
                            : r.status === "no_raw_match"
                              ? "never fired"
                              : "skipped (would over-lengthen)"}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
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
            ⌨ Shortcuts <kbd>{REVIEW_KEY.help}</kbd>
          </button>
          {/* Annotation selection toolbar (issue #86): rendered by the layer only
              while a transcript selection (or an edit) is active. Select text — or
              press `h` — to highlight it; no Copy (export is Landing 2). */}
          {annotationToolbar}
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
                {/* Deterministic domain-pack correction marker (issue #83) —
                    DISTINCT from "edited" (an operator's own change): this edit was
                    made automatically by the run's domain pack. A version mismatch
                    reads as an honest "unavailable" note instead of a rule list. */}
                {corrections?.status === "shown" && (
                  <button
                    type="button"
                    onClick={() => setProvOpen((on) => !on)}
                    aria-expanded={provOpen}
                    aria-controls="review-provenance-body"
                    className="tp-corrected-chip ml-2"
                  >
                    corrected by domain pack ({shownEntries.length}){" "}
                    {provOpen ? "▾" : "▸"}
                  </button>
                )}
                {corrections?.status === "unavailable" && (
                  <span className="muted text-sm ml-2" role="note">
                    correction provenance unavailable
                    {corrections.recordedVersion != null
                      ? ` (recorded by corrector v${corrections.recordedVersion}; this console reads a different version)`
                      : ""}
                  </span>
                )}
              </p>
              {corrections?.status === "shown" && provOpen && (
                <ul
                  id="review-provenance-body"
                  className="review-provenance-list text-sm my-1"
                >
                  {shownEntries.map((entry, i) => (
                    <li key={`${entry.id}-${i}`}>
                      {entry.resolved ? (
                        <>
                          <code>{entry.match}</code> → <code>{entry.replace}</code>{" "}
                          <span className="muted">
                            ({entry.pack} · rule <code>{entry.id}</code>)
                          </span>
                        </>
                      ) : (
                        <span className="muted">
                          unresolved rule <code>{entry.id}</code> (
                          <code>{entry.from}</code> → <code>{entry.to}</code>)
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              )}
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
                  // The chord matcher is shared with the on-screen hints and the
                  // cheat-sheet (keymap.ts) so the three can never disagree. Skip
                  // it mid-IME-composition, or an Enter that is committing a
                  // candidate (CJK and others) would save half-composed text.
                  if (isSaveEditChord(e) && !e.nativeEvent.isComposing) {
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
              {/* Immutable raw ASR text, one action away (issue #83): reveal to
                  compare the operator-effective text against what the model first
                  heard, copy it, or reset the edit box to it. Reset POPULATES THE
                  BOX ONLY — it never persists; the operator still presses Save (so
                  the unsaved-edit discard protection stays intact). Hidden when the
                  segment has no distinct raw text (split children / synthetic
                  lines send rawText: null). */}
              {current.rawText != null && (
                <div className="review-raw my-1">
                  <button
                    type="button"
                    onClick={() => setRawOpen((on) => !on)}
                    aria-expanded={rawOpen}
                    aria-controls="review-raw-body"
                    className="text-sm"
                  >
                    {rawOpen ? "▾" : "▸"} Original (raw) transcript
                  </button>
                  {rawOpen && (
                    <div id="review-raw-body" className="my-1">
                      <textarea
                        readOnly
                        value={current.rawText}
                        rows={2}
                        className="w-full text-sm"
                        aria-label="Original raw transcript text for this segment (read only)"
                      />
                      <div className="flex items-center my-1">
                        <button
                          type="button"
                          onClick={() => void copyRaw()}
                          className="mr-2 text-sm"
                        >
                          Copy raw text
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (current.rawText == null) return;
                            setEditText(current.rawText);
                            setConfirmDiscard(false);
                            editRef.current?.focus();
                          }}
                          disabled={isSplitParent}
                          className="text-sm"
                        >
                          Reset edit to raw
                        </button>
                      </div>
                      <p
                        role="status"
                        aria-live="polite"
                        className="muted text-sm"
                      >
                        {copyStatus ?? ""}
                      </p>
                    </div>
                  )}
                </div>
              )}
              <div className="flex items-center my-1">
                <button
                  type="button"
                  onClick={() => void verifyAndAdvance()}
                  disabled={busy}
                  className="primary mr-2"
                >
                  Verify &amp; next <kbd>{REVIEW_KEY.verify}</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => void saveEdit()}
                  disabled={busy || isSplitParent}
                  className="mr-2"
                >
                  Save edit <kbd>{SAVE_EDIT_LABEL}</kbd>
                </button>
                <button
                  type="button"
                  onClick={jumpNext}
                  disabled={busy}
                  className="mr-2"
                >
                  Skip <kbd>{REVIEW_KEY.skip}</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => cursor >= 0 && play(cursor)}
                  disabled={busy}
                >
                  Replay <kbd>{REVIEW_KEY.replay}</kbd>
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
                  Assign speaker
                  {speakers.length > 0 && (
                    // The digit keys only assign when the roster has slots, so the
                    // cue appears only then (honest-UX: no key advertised that no-ops).
                    <>
                      {" "}
                      (<kbd>{ASSIGN_DIGIT_MIN}</kbd>–<kbd>{ASSIGN_DIGIT_MAX}</kbd>)
                    </>
                  )}
                  :{" "}
                  <select
                    className="text-sm"
                    value=""
                    disabled={busy}
                    aria-label="Assign a speaker to this whole segment"
                    onChange={(e) => {
                      const val = e.target.value;
                      // Blur back to the document so the global keymap (suppressed
                      // while a <select> holds focus) is live again right after a
                      // mouse pick — otherwise v/j/digits would silently do nothing.
                      e.currentTarget.blur();
                      if (val === "") return;
                      void reassignSegment(val === "__inherit__" ? null : val);
                    }}
                  >
                    <option value="">Assign speaker…</option>
                    {speakers.map((sp, i) => (
                      <option key={sp.id} value={sp.id}>
                        {i < ASSIGN_DIGIT_MAX ? `${i + ASSIGN_DIGIT_MIN}. ` : ""}
                        {sp.displayName}
                      </option>
                    ))}
                    <option value="__inherit__">Reset to detected speaker</option>
                  </select>
                </label>
              )}
              {confirmDiscard && (
                <p role="alert" className="text-sm">
                  You have an unsaved edit. Press <kbd>{SAVE_EDIT_LABEL}</kbd> to save
                  it, or repeat the action (<kbd>{REVIEW_KEY.verify}</kbd> to
                  verify, or click the
                  word again to split) to discard the edit and continue.
                </p>
              )}
              {error && (
                <p role="alert" className="text-sm">
                  {error}
                </p>
              )}
              {/* Speaker-assignment success, spoken politely for a screen-reader
                  operator (the inline name change is not in a live region). */}
              <p role="status" aria-live="polite" className="visually-hidden">
                {assignStatus ?? ""}
              </p>
            </div>
          )}
            <KeymapHelp
              open={helpOpen}
              onClose={() => setHelpOpen(false)}
              hasRoster={speakers.length > 0}
            />
          </div>
        )}
      </section>
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
        // Operator annotation layer (issue #86): per-line highlight spans + stale
        // locator lines to paint, and the mouseup selection signal (only when this
        // tab holds the claim — the read-only surface gets no listener).
        annotationSpans={annotationSpans}
        staleLocators={annotationStaleLines}
        onTextSelect={writable ? annotationCapture : undefined}
      />
      {/* Highlights panel (issue #86): read for every viewer; the mutating actions
          render only when this tab holds the claim. Placed after the transcript so
          "Jump" moves the cursor into the list above. */}
      {annotationPanel}
    </div>
  );
}
