import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

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
import { partition, type LabelStateShape } from "../lib/speaker-bands";
import { useAnnotations } from "./AnnotationLayer";
import { KeymapHelp } from "./KeymapHelp";
import { OutlinePanel } from "./OutlinePanel";
import {
  ASSIGN_DIGIT_MAX,
  ASSIGN_DIGIT_MIN,
  isSaveEditChord,
  REVIEW_KEY,
  SAVE_EDIT_LABEL,
  SPEAKER_ALIAS,
} from "./keymap";
import { SpeakerAssignPopover } from "./SpeakerAssignPopover";
import { SpeakerCombobox } from "./SpeakerCombobox";
import { type LabelsResult, SpeakerRail } from "./SpeakerRail";
import { UndoToast } from "./UndoToast";
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
  renameCsrf?: string | null;
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
  speakers: initialSpeakers,
  outline,
  annotations: initialAnnotations = [],
  annotationTags: initialAnnotationTags = [],
  annotationLimits = FALLBACK_ANNOTATION_LIMITS,
  tagCsrf: initialTagCsrf = null,
  clipCsrf: initialClipCsrf = null,
  claimCsrf = null,
  renameCsrf = null,
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
  const [speakers, setSpeakers] = useState(initialSpeakers);
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
  const [undoInfo, setUndoInfo] = useState<LabelsResult["undo"] | null>(null);
  const [labelStates, setLabelStates] = useState(initialLabelStates);
  const [highlightLabels, setHighlightLabels] = useState<ReadonlySet<string>>(new Set());
  useEffect(() => {
    if (highlightLabels.size === 0) return;
    const timeout = setTimeout(() => setHighlightLabels(new Set()), 300);
    return () => clearTimeout(timeout);
  }, [highlightLabels]);
  const labelResolutions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const seg of segments) {
      if (seg.label != null) counts.set(seg.label, (counts.get(seg.label) ?? 0) + 1);
    }
    return new Map(labelStates.map((ls) => [ls.label, {
      speakerId: ls.speakerId,
      speakerName: ls.speakerName,
      resolved: ["human_assign", "human_exclude", "human_unknown", "grounded_cosine", "auto_enroll"].includes(ls.resolution),
      segmentCount: counts.get(ls.label) ?? 0,
    }]));
  }, [labelStates, segments]);
  const { needsYouLabels, unresolvedCount, affectedSegmentCount } = useMemo(() => {
    const { needsYou } = partition(labelStates);
    const needsYouLabels = new Set(needsYou.map((state) => state.label));
    let affectedSegmentCount = 0;
    for (const [label, resolution] of labelResolutions) {
      if (needsYouLabels.has(label)) {
        affectedSegmentCount += resolution.segmentCount;
      }
    }
    return { needsYouLabels, unresolvedCount: needsYou.length, affectedSegmentCount };
  }, [labelStates, labelResolutions]);
  const [popoverTarget, setPopoverTarget] = useState<{
    segmentIndex: number;
    anchorRect: DOMRect;
  } | null>(null);
  const closePopover = useCallback(() => setPopoverTarget(null), []);
  const handleSpeakerClick = useCallback((segmentIndex: number, anchorRect: DOMRect) => {
    if (busyRef.current) return;
    setPopoverTarget((prev) =>
      prev?.segmentIndex === segmentIndex ? null : { segmentIndex, anchorRect },
    );
  }, [busyRef]);
  const popoverSegment = popoverTarget ? segments[popoverTarget.segmentIndex] : undefined;
  const popoverResolution = labelResolutions.get(popoverSegment?.label ?? "");
  const popoverSpeakerId = popoverSegment
    ? popoverSegment.wordRangeSpeakerId ?? popoverResolution?.speakerId ?? null
    : null;
  // Hide rename when the segment's effective speaker differs from the label's
  // resolution (a per-segment override exists). Renaming via the label's ID
  // would silently rename a different roster speaker than the one displayed.
  const popoverCanRename =
    popoverSpeakerId != null &&
    popoverResolution?.speakerId != null &&
    popoverSegment?.speaker === popoverResolution.speakerName;

  const { cursor, setCursor, goTo, jumpNext, remaining } = useWalkCursor(
    segments,
    initialSegments,
    play,
  );
  useEffect(() => {
    setPopoverTarget(null);
  }, [cursor, writable]);
  const [pendingPopoverIndex, setPendingPopoverIndex] = useState<number | null>(null);
  useLayoutEffect(() => {
    if (pendingPopoverIndex == null || pendingPopoverIndex !== cursor) return;
    setPendingPopoverIndex(null);
    const row = playerRef.current?.focusCursorRow();
    const btn = row?.querySelector<HTMLElement>(".tp-speaker-btn");
    if (btn) {
      btn.focus();
      handleSpeakerClick(pendingPopoverIndex, btn.getBoundingClientRect());
    }
  }, [cursor, pendingPopoverIndex, handleSpeakerClick]);

  const hearableLabels = useMemo(
    () =>
      new Set(
        segments.flatMap((segment) => (segment.label ? [segment.label] : [])),
      ),
    [segments],
  );
  const hearVoice = useCallback(
    (label: string) => {
      const index = segments.findIndex((segment) => segment.label === label);
      if (index >= 0) goTo(index);
    },
    [segments, goTo],
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
      const changedLabels = new Set<string>();
      for (const newLs of result.labels) {
        const oldLs = labelStates.find((ls) => ls.label === newLs.label);
        if (!oldLs || oldLs.speakerId !== newLs.speakerId || oldLs.resolution !== newLs.resolution) {
          changedLabels.add(newLs.label);
        }
      }
      if (changedLabels.size > 0) setHighlightLabels(changedLabels);

      setSegments(result.segments);
      setProgress(result.progress);
      setLabelStates(result.labels);
      setUndoInfo(result.undo ?? null);
      if (result.speakers) {
        setSpeakers((prev) => {
          const incoming = result.speakers!;
          const incomingIds = new Set(incoming.map((s) => s.id));
          const kept = prev
            .filter((s) => incomingIds.has(s.id))
            .map((s) => {
              const fresh = incoming.find((i) => i.id === s.id);
              return fresh ? { ...s, displayName: fresh.displayName } : s;
            });
          const keptIds = new Set(kept.map((s) => s.id));
          const added = incoming.filter((s) => !keptIds.has(s.id));
          return [...kept, ...added];
        });
      }
      void reloadAnnotationsRef.current?.();
    },
    [setSegments, setProgress, labelStates],
  );

  const current =
    cursor >= 0 && cursor < segments.length ? segments[cursor] : null;
  const focusParentId = current?.sourceSegmentId ?? null;
  const isSplitParent = siblingCount(segments, focusParentId) > 1;
  const speakerDisplayName =
    current?.speaker?.trim() || current?.label?.trim() || "Unknown speaker";
  const rawLabel = current?.label?.trim() ?? "";
  const showRawLabel = rawLabel !== "" && rawLabel !== speakerDisplayName;

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
    async (speakerId: string | null, target: Segment | null = current) => {
      if (busyRef.current) return;
      const targetParentId = target?.sourceSegmentId ?? null;
      if (targetParentId === null) return;
      if (siblingCount(segments, targetParentId) > 1) {
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
          `/review/${runId}/segments/${targetParentId}/relabel`,
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
    [postForm, runId, current, segments, speakers, busyRef, setBusy],
  );

  const handlePopoverAssign = useCallback(async (speakerId: string, scope: "segment" | "label") => {
    if (!popoverTarget || !popoverSegment || !writable || busyRef.current) return;
    closePopover();
    if (scope === "segment") {
      setCursor(popoverTarget.segmentIndex);
      const isChild = popoverSegment.wordStart != null && popoverSegment.wordEnd != null;
      if (isChild) {
        await reassignChild(popoverSegment, speakerId);
      } else {
        await reassignSegment(speakerId, popoverSegment);
      }
      return;
    }
    if (!popoverSegment.label) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const result = await postForm<LabelsResult>(
        `/review/${runId}/labels/${encodeURIComponent(popoverSegment.label)}/decision`,
        { nonce: makeNonce(), action: "assign", speaker_id: speakerId },
        { claimLostOnConflict: false },
      );
      if (result) onLabelsChanged(result);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [popoverTarget, popoverSegment, writable, busyRef, closePopover, setCursor, reassignSegment, reassignChild, setBusy, postForm, runId, onLabelsChanged]);

  const handlePopoverReset = useCallback(async (scope: "segment" | "label") => {
    if (!popoverTarget || !popoverSegment || !writable || busyRef.current) return;
    closePopover();
    if (scope === "label") {
      setError("Reset is only supported for just this segment. Choose that scope to reset.");
      return;
    }
    setCursor(popoverTarget.segmentIndex);
    const isChild = popoverSegment.wordStart != null && popoverSegment.wordEnd != null;
    if (isChild) {
      await reassignChild(popoverSegment, null);
    } else {
      await reassignSegment(null, popoverSegment);
    }
  }, [popoverTarget, popoverSegment, writable, busyRef, closePopover, setCursor, reassignSegment, reassignChild]);

  const handlePopoverCreate = useCallback(async (name: string): Promise<boolean> => {
    if (!popoverSegment?.label || !writable || busyRef.current) return false;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      // V1 enrollment always assigns the entire label, regardless of selected scope.
      const result = await postForm<LabelsResult>(
        `/review/${runId}/labels/${encodeURIComponent(popoverSegment.label)}/enroll`,
        { nonce: makeNonce(), display_name: name },
        { claimLostOnConflict: false },
      );
      if (!result) return false;
      onLabelsChanged(result);
      return true;
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [popoverSegment, writable, busyRef, setBusy, postForm, runId, onLabelsChanged]);

  const handlePopoverRename = useCallback(async (newName: string) => {
    if (!popoverSpeakerId || !popoverSegment || !writable || busyRef.current) return;
    closePopover();
    if (!renameCsrf) {
      setError("Could not rename speaker: reload the editor to refresh the security token.");
      return;
    }
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const res = await apiFetch(`/speakers/${popoverSpeakerId}/rename`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded", accept: "application/json" },
        body: new URLSearchParams({ display_name: newName, csrf_token: renameCsrf }).toString(),
      });
      const data = await res.json() as { id: string; displayName: string };
      setSpeakers((prev) => prev.map((speaker) =>
        speaker.id === data.id ? { ...speaker, displayName: data.displayName } : speaker));
      const renamedLabel = popoverSegment.label;
      setSegments((prev) => prev.map((seg) =>
        seg.label === renamedLabel && seg.speaker === popoverSegment.speaker
          ? { ...seg, speaker: data.displayName } : seg));
      setLabelStates((prev) => prev.map((ls) => ({
        ...ls,
        speakerName: ls.speakerId === data.id ? data.displayName : ls.speakerName,
        cosineSpeakerName: ls.cosineSpeakerId === data.id ? data.displayName : ls.cosineSpeakerName,
        candidateSpeakerName: ls.candidateSpeakerId === data.id ? data.displayName : ls.candidateSpeakerName,
      })));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not rename speaker.");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [popoverSpeakerId, popoverSegment, writable, busyRef, closePopover, renameCsrf, setBusy]);

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
        case REVIEW_KEY.speaker:
        case SPEAKER_ALIAS: {
          event.preventDefault();
          const row = playerRef.current?.focusCursorRow();
          const btn = row?.querySelector<HTMLElement>(".tp-speaker-btn");
          if (btn) {
            btn.focus();
            handleSpeakerClick(cursor, btn.getBoundingClientRect());
          }
          break;
        }
        case REVIEW_KEY.resetSpeaker:
          event.preventDefault();
          void reassignSegment(null);
          break;
        case REVIEW_KEY.sameAsPrevious: {
          event.preventDefault();
          if (cursor <= 0) break;
          const prevSeg = segments[cursor - 1];
          if (!prevSeg?.label) {
            setAssignStatus("Previous segment has no speaker to copy.");
            break;
          }
          const prevResolution = labelResolutions.get(prevSeg.label);
          const prevSpeakerId = prevSeg.wordRangeSpeakerId ?? prevResolution?.speakerId ?? null;
          if (!prevSpeakerId) {
            setAssignStatus("Previous segment has no assigned speaker.");
            break;
          }
          const target = segments[cursor];
          if (target && target.wordStart != null && target.wordEnd != null) {
            void reassignChild(target, prevSpeakerId);
          } else {
            void reassignSegment(prevSpeakerId);
          }
          break;
        }
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
    reassignChild,
    labelResolutions,
    speakers,
    annotateHotkey,
    handleSpeakerClick,
  ]);

  // Download shortcut: separate from the writable-gated handler so it works
  // for read-only visitors too.
  useEffect(() => {
    const onExportKey = (event: KeyboardEvent) => {
      if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
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
      if (event.key.toLowerCase() !== REVIEW_KEY.download) return;
      event.preventDefault();
      const menu = document.getElementById("export-menu");
      if (!menu) return;
      const details = menu.querySelector("details");
      if (details) {
        details.open = !details.open;
        const summary = details.querySelector("summary");
        if (summary) summary.focus({ preventScroll: true });
      }
      menu.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };
    window.addEventListener("keydown", onExportKey);
    return () => window.removeEventListener("keydown", onExportKey);
  }, []);

  const done = progress.total > 0 && remaining === 0;

  return (
    <>
      <div
        ref={annotationRootRef}
        className={writable && walkMode ? "walk-active" : undefined}
      >
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
            ? `Cursor on segment at ${current.start.toFixed(1)} seconds, speaker ${speakerDisplayName}${current.verified ? ", verified" : ""}${current.corrected ? ", edited" : ""}`
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
                className={
                  walkMode ? "me-walk-btn-active text-sm" : "text-sm"
                }
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

        {writable && unresolvedCount > 0 && (
          <div
            className="me-unresolved-banner"
            role="status"
            aria-live="polite"
            style={{
              padding: "0.5rem 0.75rem",
              background: "var(--surface-2)",
              color: "var(--ink-muted)",
              fontSize: "0.875rem",
            }}
          >
            {unresolvedCount} unidentified {unresolvedCount === 1 ? "voice" : "voices"},{" "}
            {affectedSegmentCount} {affectedSegmentCount === 1 ? "segment" : "segments"} affected ·{" "}
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                if (busyRef.current) return;
                const index = segments.findIndex(
                  (segment) => segment.label != null && needsYouLabels.has(segment.label),
                );
                if (index < 0) return;
                goTo(index);
                setPendingPopoverIndex(index);
              }}
              style={{
                padding: 0,
                border: 0,
                background: "none",
                color: "var(--accent)",
                font: "inherit",
                textDecoration: "underline",
                cursor: "pointer",
              }}
            >
              Start reviewing
            </button>
          </div>
        )}

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
                <p className="me-segment-heading muted text-sm">
                  <span
                    className={`me-speaker-identity${current.paletteIndex != null ? ` spk-${current.paletteIndex}` : ""}`}
                  >
                    <strong>{speakerDisplayName}</strong>
                    {showRawLabel && (
                      <span className="spk-badge">{rawLabel}</span>
                    )}
                  </span>
                  <span>
                    {walkMode ? "Walk" : "Editing"} segment at{" "}
                    {current.start.toFixed(2)}s
                  </span>
                  {current.confidence != null &&
                    current.confidence < lowConfidenceThreshold && (
                      <span className="tp-uncertain-chip">uncertain</span>
                    )}
                  {current.verified && (
                    <span className="spk-badge">verified</span>
                  )}
                  {current.corrected && (
                    <span className="spk-badge">edited</span>
                  )}
                  {current.corrections?.status === "shown" && (
                    <button
                      type="button"
                      onClick={() => setProvOpen((on) => !on)}
                      aria-expanded={provOpen}
                      aria-controls="editor-provenance-body"
                      className="tp-corrected-chip"
                    >
                      corrected by domain pack (
                      {current.corrections.entries.length}) {provOpen ? "▾" : "▸"}
                    </button>
                  )}
                  {current.corrections?.status === "unavailable" && (
                    <span className="muted text-sm" role="note">
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
                  <div className="tp-reassign text-sm my-1">
                    Assign speaker
                    {speakers.length > 0 && (
                      <>
                        {" "}
                        (<kbd>{ASSIGN_DIGIT_MIN}</kbd>–
                        <kbd>{ASSIGN_DIGIT_MAX}</kbd>)
                      </>
                    )}
                    :{" "}
                    <SpeakerCombobox
                      speakers={speakers}
                      mode="command"
                      label="Assign a speaker to this whole segment"
                      placeholder="Assign speaker…"
                      disabled={busy}
                      digitPrefixes
                      onSelect={(id) => void reassignSegment(id)}
                      onInherit={() => void reassignSegment(null)}
                    />
                  </div>
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
              onSpeakerClick={writable ? handleSpeakerClick : undefined}
              popoverSegmentIndex={popoverTarget?.segmentIndex ?? null}
              labelResolutions={labelResolutions}
              highlightLabels={highlightLabels}
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
            {writable && popoverTarget && popoverSegment && (
              <SpeakerAssignPopover
                key={popoverTarget.segmentIndex}
                anchorRect={popoverTarget.anchorRect}
                currentSpeaker={popoverSegment.speaker}
                currentSpeakerId={popoverCanRename ? popoverSpeakerId : null}
                currentLabel={popoverSegment.label}
                resolved={popoverResolution?.resolved ?? false}
                labelSegmentCount={popoverResolution?.segmentCount ?? 1}
                speakers={speakers}
                onAssign={handlePopoverAssign}
                onCreate={handlePopoverCreate}
                onReset={handlePopoverReset}
                onRename={handlePopoverRename}
                onClose={closePopover}
                disabled={busy || !writable}
              />
            )}
            {annotationPanel}
          </div>

          <SpeakerRail
            runId={runId}
            reviewToken={reviewToken}
            writable={writable}
            labelStates={labelStates}
            speakers={speakers}
            onClaimLost={onAnnotationClaimLost}
            onLabelsChanged={onLabelsChanged}
            onHearVoice={capability.seekEnabled ? hearVoice : undefined}
            hearableLabels={hearableLabels}
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
      {undoInfo && reviewToken && (
        <UndoToast
          undo={undoInfo}
          runId={runId}
          reviewToken={reviewToken}
          claimCsrf={claimCsrf}
          onClaimLost={onAnnotationClaimLost}
          onUndone={(data) => {
            setUndoInfo(null);
            onLabelsChanged(data);
          }}
          onDismiss={() => setUndoInfo(null)}
        />
      )}
    </>
  );
}
