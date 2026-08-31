import { useCallback, useEffect, useRef, useState } from "react";

import {
  FALLBACK_ANNOTATION_LIMITS,
  type AnnotationLimits,
  type AnnotationShape,
  type AnnotationTagShape,
} from "../lib/annotations";
import { ApiError, apiFetch } from "../lib/api-client";
import {
  type SegmentPatchResult,
  nextTarget,
  siblingCount,
  useBusyGuard,
  useFormPost,
  useSegmentPatch,
  useWalkCursor,
} from "../lib/editor-mutations";
import { makeNonce } from "../lib/nonce";
import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import type { OutlineProps } from "../lib/outline";
import { useAnnotations } from "./AnnotationLayer";
import { KeymapHelp } from "./KeymapHelp";
import { OutlinePanel } from "./OutlinePanel";
import {
  ASSIGN_DIGIT_MAX,
  ASSIGN_DIGIT_MIN,
  isSaveEditChord,
  REVIEW_KEY,
  SAVE_EDIT_LABEL,
} from "./keymap";
import { type LabelStateShape, type LabelsResult, SpeakerRail } from "./SpeakerRail";
import {
  type Segment,
  type SplitWord,
  TranscriptPlayer,
  type TranscriptPlayerHandle,
} from "./TranscriptPlayer";

export interface MediaEditorProps {
  mediaId: string;
  runId: string;
  mediaUrl: string;
  segments: Segment[];
  capability: PlaybackCapability;
  lowConfidenceThreshold: number;
  reviewToken: string | null;
  initialProgress: { verified: number; total: number };
  peaksUrl?: string | null;
  turns?: Turn[];
  speakers: { id: string; displayName: string }[];
  outline?: OutlineProps;
  annotations?: AnnotationShape[];
  annotationTags?: AnnotationTagShape[];
  annotationLimits?: AnnotationLimits;
  tagCsrf?: string | null;
  clipCsrf?: string | null;
  claimCsrf?: string | null;
  multiUser?: boolean;
  labelStates?: LabelStateShape[];
  translate?: {
    csrf: string;
    defaultTarget: string | null;
    defaultTargetLabel: string | null;
    active: boolean;
    runAnchor: string;
    transcriptUrl: string;
  } | null;
}


export function MediaEditor({
  mediaId,
  runId,
  mediaUrl,
  segments: initialSegments,
  capability,
  lowConfidenceThreshold,
  reviewToken: initialReviewToken,
  initialProgress,
  peaksUrl,
  turns,
  speakers,
  outline,
  annotations: initialAnnotations = [],
  annotationTags: initialAnnotationTags = [],
  annotationLimits = FALLBACK_ANNOTATION_LIMITS,
  tagCsrf: initialTagCsrf = null,
  clipCsrf: initialClipCsrf = null,
  claimCsrf = null,
  multiUser = false,
  labelStates: initialLabelStates = [],
  translate = null,
}: MediaEditorProps): React.JSX.Element {
  const [segments, setSegments] = useState<Segment[]>(initialSegments);
  const [progress, setProgress] = useState(initialProgress);
  const [editText, setEditText] = useState("");
  const { busy, busyRef, setBusy } = useBusyGuard();
  const [reviewToken, setReviewToken] = useState<string | null>(
    initialReviewToken,
  );
  const [tagCsrf, setTagCsrf] = useState(initialTagCsrf);
  const [clipCsrf, setClipCsrf] = useState(initialClipCsrf);
  const [claimLost, setClaimLost] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [walkMode, setWalkMode] = useState(
    () => initialProgress.verified < initialProgress.total,
  );
  const [splitMode, setSplitMode] = useState(false);
  const [splitData, setSplitData] = useState<{
    segmentId: string;
    splittable: boolean;
    reason: string | null;
    words: SplitWord[];
  } | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const helpOpenRef = useRef(false);
  helpOpenRef.current = helpOpen;
  const [assignStatus, setAssignStatus] = useState<string | null>(null);
  const [provOpen, setProvOpen] = useState(false);
  const [translatePhase, setTranslatePhase] = useState<
    "idle" | "starting" | "started" | "error"
  >(translate?.active ? "started" : "idle");
  const [translateError, setTranslateError] = useState<string | null>(null);
  const translateBusyRef = useRef(false);
  const [claiming, setClaiming] = useState(false);
  const claimingRef = useRef(false);
  const reviewTokenRef = useRef(reviewToken);
  reviewTokenRef.current = reviewToken;

  const claimForEditing = useCallback(async () => {
    if (!claimCsrf || claimingRef.current) return;
    claimingRef.current = true;
    setClaiming(true);
    try {
      const body = new URLSearchParams({
        run_id: runId,
        csrf_token: claimCsrf,
      });
      const res = await apiFetch(`/media/${mediaId}/editor/claim`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: body.toString(),
      });
      const data = (await res.json()) as {
        token: string;
        tagCsrf: string;
        clipCsrf: string;
      };
      reviewTokenRef.current = data.token;
      setReviewToken(data.token);
      setTagCsrf(data.tagCsrf);
      setClipCsrf(data.clipCsrf);
      setClaimLost(false);
      const p = new URLSearchParams(window.location.search);
      p.set("token", data.token);
      window.history.replaceState(null, "", `?${p}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not claim.");
    } finally {
      claimingRef.current = false;
      setClaiming(false);
    }
  }, [claimCsrf, runId, mediaId]);

  const claimed = reviewToken !== null && !claimLost;

  // Heartbeat: refresh the claim TTL without rotating the token.  A stale
  // tab whose token no longer matches gets 409 and drops to claimLost.
  // The interval keeps running in background tabs (browsers throttle to
  // ~1/min, well within the default 600s TTL).  An immediate catch-up
  // refresh fires when the tab becomes visible again.
  useEffect(() => {
    if (!claimed || !claimCsrf) return;

    const doRefresh = async () => {
      const tok = reviewTokenRef.current;
      if (!tok) return;
      try {
        const body = new URLSearchParams({
          run_id: runId,
          token: tok,
          csrf_token: claimCsrf,
        });
        await apiFetch(`/media/${mediaId}/editor/refresh`, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
          body: body.toString(),
        });
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          setClaimLost(true);
        }
      }
    };

    const intervalId = setInterval(() => void doRefresh(), 60_000);

    const onVisibility = () => {
      if (document.visibilityState === "visible") void doRefresh();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      clearInterval(intervalId);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [claimed, claimCsrf, runId, mediaId]);

  // Single-operator auto-claim on mount: skip the manual "Claim for editing"
  // click when there is only one operator and no handoff token is present.
  useEffect(() => {
    if (multiUser || initialReviewToken || !claimCsrf) return;
    void claimForEditing();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only
  }, []);

  // Release on unload (best-effort — sendBeacon for reliability).
  useEffect(() => {
    if (!claimed) return;
    const onUnload = () => {
      const tok = reviewTokenRef.current;
      if (!tok) return;
      const body = new URLSearchParams({
        run_id: runId,
        token: tok,
      });
      navigator.sendBeacon(
        `/media/${mediaId}/editor/release`,
        body,
      );
    };
    window.addEventListener("beforeunload", onUnload);
    window.addEventListener("pagehide", onUnload);
    return () => {
      window.removeEventListener("beforeunload", onUnload);
      window.removeEventListener("pagehide", onUnload);
    };
  }, [claimed, runId, mediaId]);

  const editTextRef = useRef(editText);
  editTextRef.current = editText;
  const confirmDiscardRef = useRef(confirmDiscard);
  confirmDiscardRef.current = confirmDiscard;

  const playerRef = useRef<TranscriptPlayerHandle>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);
  const annotationRootRef = useRef<HTMLDivElement>(null);
  const reloadAnnotationsRef = useRef<(() => Promise<void>) | null>(null);

  const play = useCallback(
    (index: number) => playerRef.current?.playSegment(index),
    [],
  );

  const writable = reviewToken !== null && !claimLost;

  const { cursor, setCursor, goTo, jumpNext, remaining } = useWalkCursor(
    segments,
    initialSegments,
    play,
  );

  const postForm = useFormPost(
    reviewToken,
    writable,
    () => setClaimLost(true),
    setError,
  );
  const applyResult = useSegmentPatch(segments, setSegments, setProgress);

  const onLabelsChanged = useCallback(
    (result: LabelsResult) => {
      setSegments(result.segments);
      setProgress(result.progress);
      void reloadAnnotationsRef.current?.();
    },
    [setSegments, setProgress],
  );

  const current =
    cursor >= 0 && cursor < segments.length ? segments[cursor] : null;
  const focusParentId = current?.sourceSegmentId ?? null;
  const isSplitParent = siblingCount(segments, focusParentId) > 1;

  useEffect(() => {
    setEditText(current?.text ?? "");
    setConfirmDiscard(false);
    setAssignStatus(null);
    setProvOpen(false);
  }, [current?.segmentId, current?.text]);

  // Focus the cursor row only after KEYBOARD-driven navigation (not pointer
  // clicks, which should leave focus on the control the user operated).
  const keyboardNavRef = useRef(false);
  const prevCursorRef = useRef(cursor);
  useEffect(() => {
    if (cursor === prevCursorRef.current) return;
    prevCursorRef.current = cursor;
    if (!keyboardNavRef.current) return;
    keyboardNavRef.current = false;
    if (claimLost || confirmDiscardRef.current) return;
    if (document.activeElement === editRef.current) return;
    const frameId = requestAnimationFrame(() => {
      if (claimLost) return;
      playerRef.current?.focusCursorRow();
    });
    return () => cancelAnimationFrame(frameId);
  }, [cursor, claimLost]);

  const postJson = useCallback(
    (
      path: string,
      body: Record<string, string>,
    ): Promise<SegmentPatchResult | null> =>
      postForm<SegmentPatchResult>(path, body),
    [postForm],
  );

  const verifyAndAdvance = useCallback(async () => {
    if (current?.segmentId == null || busyRef.current) return;
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
      if (walkMode) {
        const next = nextTarget(patched, index + 1);
        if (next >= 0) {
          keyboardNavRef.current = true;
          goTo(next);
        }
      }
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
    walkMode,
    busyRef,
    setBusy,
  ]);

  const saveEdit = useCallback(async () => {
    if (current?.segmentId == null || busyRef.current) return;
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
      void reloadAnnotationsRef.current?.();
      editRef.current?.blur();
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [current, cursor, postJson, runId, editText, applyResult, segments, busyRef, setBusy]);

  const currentRef = useRef(current);
  currentRef.current = current;

  const splitAt = useCallback(
    async (sourceSegmentId: string, wordIndex: number) => {
      if (busyRef.current) return;
      const cur = currentRef.current;
      if (cur && editTextRef.current !== (cur.text ?? "") && !confirmDiscardRef.current) {
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
          { claimLostOnConflict: false },
        );
        if (!result) return;
        setConfirmDiscard(false);
        setSegments(result.segments);
        setProgress(result.progress);
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
    [postForm, runId, busyRef, setBusy, setCursor],
  );

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
        void reloadAnnotationsRef.current?.();
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [postForm, runId, busyRef, setBusy],
  );

  const reassignSegment = useCallback(
    async (speakerId: string | null) => {
      if (busyRef.current) return;
      if (focusParentId === null) return;
      if (isSplitParent) {
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
        setSegments(result.segments);
        setProgress(result.progress);
        void reloadAnnotationsRef.current?.();
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
    [postForm, runId, focusParentId, isSplitParent, speakers, busyRef, setBusy],
  );

  // Lazily fetch split words when split mode is engaged.
  useEffect(() => {
    if (!splitMode || !writable || focusParentId === null || isSplitParent) {
      setSplitData(null);
      return;
    }
    const parentId = focusParentId;
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
          setSplitData({
            segmentId: parentId,
            splittable: data.splittable,
            reason: data.reason,
            words: data.words,
          });
        }
      } catch (err) {
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
  }, [splitMode, writable, focusParentId, isSplitParent, runId, current?.corrected]);

  const startTranslate = useCallback(async () => {
    if (translateBusyRef.current) return;
    if (!translate || !translate.defaultTarget) return;
    translateBusyRef.current = true;
    setTranslatePhase("starting");
    try {
      const body = new URLSearchParams({
        csrf_token: translate.csrf,
        target_language: translate.defaultTarget,
      });
      const res = await apiFetch(`/runs/${runId}/translation/generate`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body,
      });
      const data = (await res.json()) as {
        started: boolean;
        error: string | null;
      };
      if (data.started) {
        setTranslatePhase("started");
      } else {
        setTranslateError(data.error ?? "Translation could not start.");
        setTranslatePhase("error");
      }
    } catch (err) {
      setTranslateError(
        err instanceof ApiError ? err.detail : "Translation could not start.",
      );
      setTranslatePhase("error");
    } finally {
      translateBusyRef.current = false;
    }
  }, [translate, runId]);

  const onAnnotationClaimLost = useCallback(() => setClaimLost(true), []);

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
    clipCsrf,
    onJump: goTo,
    onClaimLost: onAnnotationClaimLost,
  });
  useEffect(() => {
    reloadAnnotationsRef.current = reloadAnnotations;
  }, [reloadAnnotations]);

  // Global keymap — same pattern as ReviewStepper, plus `w` for walk-mode toggle.
  useEffect(() => {
    if (!writable) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (helpOpenRef.current) return;
      const el = event.target as HTMLElement | null;
      const tag = el?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el?.isContentEditable
      )
        return;
      switch (event.key.toLowerCase()) {
        case REVIEW_KEY.verify:
          event.preventDefault();
          void verifyAndAdvance();
          break;
        case REVIEW_KEY.skip:
          event.preventDefault();
          keyboardNavRef.current = true;
          jumpNext();
          setTimeout(() => { keyboardNavRef.current = false; }, 0);
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
          event.preventDefault();
          const next = Math.min(cursor + 1, segments.length - 1);
          if (next !== cursor) {
            keyboardNavRef.current = true;
            goTo(next);
          }
          break;
        }
        case REVIEW_KEY.previous: {
          event.preventDefault();
          const prev = Math.max(cursor - 1, 0);
          if (prev !== cursor) {
            keyboardNavRef.current = true;
            goTo(prev);
          }
          break;
        }
        case REVIEW_KEY.resetSpeaker:
          event.preventDefault();
          void reassignSegment(null);
          break;
        case REVIEW_KEY.help:
          event.preventDefault();
          setHelpOpen(true);
          break;
        case REVIEW_KEY.annotate:
          event.preventDefault();
          annotateHotkey();
          break;
        case REVIEW_KEY.walkMode:
          event.preventDefault();
          setWalkMode((on) => !on);
          break;
        default:
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

  const done = progress.total > 0 && remaining === 0;

  return (
    <>
      <div ref={annotationRootRef}>
        {claimLost && (
          <p role="alert" className="notice text-sm">
            Your claim expired or was taken over. Everything you already saved is
            safe. Copy any unsaved edit from the box below, then{" "}
            <button
              type="button"
              onClick={() => void claimForEditing()}
              disabled={claiming}
              className="underline"
            >
              {claiming ? "Re-claiming…" : "re-claim to continue editing"}
            </button>
            .
          </p>
        )}
        {!reviewToken && !claimLost && claimCsrf && (
          <div className="notice text-sm">
            Read-only view.{" "}
            <button
              type="button"
              onClick={() => void claimForEditing()}
              disabled={claiming}
              className="btn-primary"
            >
              {claiming ? "Claiming…" : "Claim for editing"}
            </button>
          </div>
        )}
        {error && !writable && (
          <p role="alert" className="notice text-sm">
            {error}
          </p>
        )}

        <p
          className="visually-hidden"
          aria-live="polite"
          aria-atomic="true"
        >
          {writable && current
            ? `Cursor on segment at ${current.start.toFixed(1)} seconds, speaker ${current.speaker}${current.verified ? ", verified" : ""}${current.corrected ? ", edited" : ""}`
            : ""}
        </p>

        <section className="me-toolbar" aria-label="Editor controls">
          <div className="progress-wrap">
            <p aria-live="polite" aria-atomic="true">
              <strong>{progress.verified}</strong> of{" "}
              <strong>{progress.total}</strong> segments verified
              {done ? ". You have checked every line." : ` · ${remaining} left`}
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
            <div className="me-actions">
              <button
                type="button"
                onClick={() => setWalkMode((on) => !on)}
                aria-pressed={walkMode}
                className="text-sm"
              >
                {walkMode ? "Exit walk mode" : "Walk mode"}{" "}
                <kbd>{REVIEW_KEY.walkMode}</kbd>
              </button>
              <button
                type="button"
                onClick={() => setSplitMode((on) => !on)}
                aria-pressed={splitMode}
                className="text-sm"
              >
                {splitMode ? "Exit split mode" : "Split"}
              </button>
              <button
                type="button"
                onClick={() => setHelpOpen(true)}
                aria-haspopup="dialog"
                className="text-sm"
              >
                Shortcuts <kbd>{REVIEW_KEY.help}</kbd>
              </button>
              {annotationToolbar}
            </div>
          )}
          {translate &&
            (translatePhase === "idle" ? (
              translate.defaultTarget ? (
                <button
                  type="button"
                  onClick={() => void startTranslate()}
                  className="text-sm"
                >
                  Translate to {translate.defaultTargetLabel}
                </button>
              ) : (
                <a href={translate.runAnchor} className="text-sm">
                  Translate this recording
                </a>
              )
            ) : translatePhase === "starting" ? (
              <span className="muted text-sm">Starting translation…</span>
            ) : translatePhase === "started" ? (
              <span className="muted text-sm">
                A translation is queued or running; the result appears on the{" "}
                <a href={translate.transcriptUrl}>transcript page</a>.
              </span>
            ) : (
              <span role="alert" className="text-sm">
                {translateError}{" "}
                <a href={translate.runAnchor}>Open the run page</a>
              </span>
            ))}
        </section>

        {done && (
          <div className="review-done card-actions">
            <a className="btn-primary" href="#export-menu">
              Download transcript
            </a>
            <a href="/media">Back to the library</a>
          </div>
        )}

        <div className="me-layout">
          <div className="lib-main">
            {writable && current && current.segmentId !== null && (
              <div className="me-segment-actions">
                <p className="muted text-sm">
                  {walkMode ? "Walk" : "Editing"} segment at{" "}
                  {current.start.toFixed(2)}s
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
                  {current.corrections?.status === "shown" && (
                    <button
                      type="button"
                      onClick={() => setProvOpen((on) => !on)}
                      aria-expanded={provOpen}
                      aria-controls="editor-provenance-body"
                      className="tp-corrected-chip ml-2"
                    >
                      corrected by domain pack (
                      {current.corrections.entries.length}) {provOpen ? "▾" : "▸"}
                    </button>
                  )}
                  {current.corrections?.status === "unavailable" && (
                    <span className="muted text-sm ml-2" role="note">
                      correction provenance unavailable
                      {current.corrections.recordedVersion != null
                        ? ` (recorded by corrector v${current.corrections.recordedVersion}; this console reads a different version)`
                        : ""}
                    </span>
                  )}
                </p>
                {current.corrections?.status === "shown" && (
                  <ul
                    id="editor-provenance-body"
                    hidden={!provOpen}
                    className="review-provenance-list text-sm my-1"
                  >
                    {current.corrections.entries.map((entry, i) => (
                      <li key={`${entry.id}-${i}`}>
                        {entry.resolved ? (
                          <>
                            <code>{entry.match}</code> →{" "}
                            <code>{entry.replace}</code>{" "}
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
                    This segment is split, so editing is disabled.
                  </p>
                )}
                {splitMode && !isSplitParent && (
                  <p className="text-sm" role="status">
                    {splitData == null
                      ? "Loading words…"
                      : splitData.splittable
                        ? "Split mode on — click a word to cut the segment before it."
                        : (splitData.reason ??
                          "This segment can't be split at a word boundary.")}
                  </p>
                )}
                <div className="flex items-center my-1">
                  <button
                    type="button"
                    onClick={() => void verifyAndAdvance()}
                    disabled={busy}
                    className="primary mr-2"
                  >
                    {walkMode ? "Verify & next" : "Verify"}{" "}
                    <kbd>{REVIEW_KEY.verify}</kbd>
                  </button>
                  <button
                    type="button"
                    onClick={() => void saveEdit()}
                    disabled={busy || isSplitParent}
                    className="mr-2"
                  >
                    Save edit <kbd>{SAVE_EDIT_LABEL}</kbd>
                  </button>
                  {walkMode && (
                    <button
                      type="button"
                      onClick={jumpNext}
                      disabled={busy}
                      className="mr-2"
                    >
                      Skip <kbd>{REVIEW_KEY.skip}</kbd>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => cursor >= 0 && play(cursor)}
                    disabled={busy}
                  >
                    Replay <kbd>{REVIEW_KEY.replay}</kbd>
                  </button>
                </div>
                {!isSplitParent && (
                  <label className="tp-reassign text-sm my-1 block">
                    Assign speaker
                    {speakers.length > 0 && (
                      <>
                        {" "}
                        (<kbd>{ASSIGN_DIGIT_MIN}</kbd>–
                        <kbd>{ASSIGN_DIGIT_MAX}</kbd>)
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
                        e.currentTarget.blur();
                        if (val === "") return;
                        void reassignSegment(
                          val === "__inherit__" ? null : val,
                        );
                      }}
                    >
                      <option value="">Assign speaker…</option>
                      {speakers.map((sp, i) => (
                        <option key={sp.id} value={sp.id}>
                          {i < ASSIGN_DIGIT_MAX
                            ? `${i + ASSIGN_DIGIT_MIN}. `
                            : ""}
                          {sp.displayName}
                        </option>
                      ))}
                      <option value="__inherit__">
                        Reset to detected speaker
                      </option>
                    </select>
                  </label>
                )}
                {confirmDiscard && (
                  <p role="alert" className="text-sm">
                    You have an unsaved edit. Press{" "}
                    <kbd>{SAVE_EDIT_LABEL}</kbd> to save it, or repeat the
                    action to discard the edit and continue.
                  </p>
                )}
                {error && (
                  <p role="alert" className="text-sm">
                    {error}
                  </p>
                )}
                <p
                  role="status"
                  aria-live="polite"
                  className="visually-hidden"
                >
                  {assignStatus ?? ""}
                </p>
              </div>
            )}
            <TranscriptPlayer
              ref={playerRef}
              runId={runId}
              mediaUrl={mediaUrl}
              segments={segments}
              capability={capability}
              lowConfidenceThreshold={lowConfidenceThreshold}
              onSegmentSelect={writable ? setCursor : undefined}
              peaksUrl={peaksUrl}
              turns={turns}
              cursorIndex={writable ? cursor : undefined}
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
              onSplitAt={writable ? splitAt : undefined}
              reassignSpeakers={writable ? speakers : undefined}
              onReassign={writable ? reassignChild : undefined}
              reassignBusy={busy}
              annotationSpans={annotationSpans}
              staleLocators={annotationStaleLines}
              onTextSelect={writable ? annotationCapture : undefined}
            />
            {annotationPanel}
          </div>

          <SpeakerRail
            runId={runId}
            reviewToken={reviewToken}
            writable={writable}
            labelStates={initialLabelStates}
            speakers={speakers}
            onClaimLost={onAnnotationClaimLost}
            onLabelsChanged={onLabelsChanged}
          />
        </div>

        <OutlinePanel
          outline={outline}
          segments={segments}
          capability={capability}
          onJump={goTo}
        />

        <KeymapHelp
          open={helpOpen}
          onClose={() => setHelpOpen(false)}
          hasRoster={speakers.length > 0}
          extraShortcuts={[
            {
              keys: REVIEW_KEY.walkMode,
              desc: "Toggle walk mode (auto-advance after verify)",
            },
          ]}
        />
      </div>
    </>
  );
}
