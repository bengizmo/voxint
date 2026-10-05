import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  FALLBACK_ANNOTATION_LIMITS,
  type AnnotationLimits,
  type AnnotationShape,
  type AnnotationTagShape,
} from "../lib/annotations";
import { createVoiceSamplePlayer, type VoiceSamplePlayer } from "../lib/voice-sample";
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
import { useEnrichmentPolling } from "../lib/enrichment-polling";
import { useWordMarks } from "../lib/use-word-marks";
import { emissionKey, marksClearedNotice, staleMarksByLine, toggleWordMark, wordMarksByLine,
  type WordMarkAction, type WordMarkUnit, type WordMarksResult } from "../lib/word-marks";
import { makeNonce } from "../lib/nonce";
import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import type { OutlineProps } from "../lib/outline";
import { partition, type LabelStateShape } from "../lib/speaker-bands";
import { useAnnotations } from "./AnnotationLayer";
import { KeymapHelp } from "./KeymapHelp";
import { OutlinePanel } from "./OutlinePanel";
import type { AssetControlsProps } from "./AssetControls";
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
import { findMergeCandidates, type MergeSuggestion } from "../lib/merge-candidates";
import { MergeSuggestionToast } from "./MergeSuggestionToast";
import {
  type Segment,
  type SplitWord,
  TranscriptPlayer,
  type TranscriptPlayerHandle,
} from "./TranscriptPlayer";

// How long the refetch after a refused undo may hold the editor write guard.
const UNDO_REFETCH_TIMEOUT_MS = 10_000;

// The whole-run reconcile a segment relabel returns; a fresh ruling carries
// its undo (issue #573).
type RelabelResult = Pick<LabelsResult, "segments" | "progress" | "undo">;

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
  assetControls?: AssetControlsProps;
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
    hasTranslation: boolean;
    activeJobId: string | null;
    csrfCancel: string | null;
    runAnchor: string;
    transcriptUrl: string;
    languageOptions: { code: string; name: string }[];
    detectedLanguage: string | null;
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
  assetControls,
}: MediaEditorProps): React.JSX.Element {
  const [segments, setSegments] = useState<Segment[]>(initialSegments);
  const [progress, setProgress] = useState(initialProgress);
  const [editText, setEditText] = useState("");
  // The current line got new text while the edit box held an unsaved edit.
  const [editOvertaken, setEditOvertaken] = useState(false);
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null);
  const { busy, busyRef, setBusy } = useBusyGuard();
  const [reviewToken, setReviewToken] = useState<string | null>(
    initialReviewToken,
  );
  const [tagCsrf, setTagCsrf] = useState(initialTagCsrf);
  const [clipCsrf, setClipCsrf] = useState(initialClipCsrf);
  const [claimLost, setClaimLost] = useState(false);
  const [speakers, setSpeakers] = useState(initialSpeakers);
  const [error, setError] = useState<string | null>(null);
  const { snapshot: marksSnapshot, refresh: refreshMarks, requestFocus: requestMarksFocus,
    adopt: adoptMarks, beginWrite: beginMarksWrite, finishWrite: finishMarksWrite,
    loadError: marksLoadError } = useWordMarks(runId, setError);
  const refreshMarksRef = useRef(refreshMarks);
  refreshMarksRef.current = refreshMarks;
  const [cleanupMode, setCleanupMode] = useState(false);
  const [cleanupNotice, setCleanupNotice] = useState<string | null>(null);
  const [selectedUnit, setSelectedUnit] = useState<{ line: string; start: number; end: number } | null>(null);
  const markUndoFocus = useRef<string | null>(null);
  const markUndoRequest = useRef<number | undefined>(undefined);
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
  const pendingOpenRef = useRef(false);
  const helpOpenRef = useRef(false);
  helpOpenRef.current = helpOpen;
  const [assignStatus, setAssignStatus] = useState<string | null>(null);
  const [provOpen, setProvOpen] = useState(false);
  const [translatePhase, setTranslatePhase] = useState<
    "idle" | "starting" | "started" | "error"
  >(translate?.active ? "started" : "idle");
  const [translateTarget, setTranslateTarget] = useState<string>(
    translate?.defaultTarget ?? "",
  );
  const [translateError, setTranslateError] = useState<string | null>(null);
  const translateBusyRef = useRef(false);
  const [assetPollTrigger, setAssetPollTrigger] = useState(false);
  const [claiming, setClaiming] = useState(false);
  const claimingRef = useRef(false);
  const reviewTokenRef = useRef(reviewToken);
  reviewTokenRef.current = reviewToken;

  const [assetPollActive, setAssetPollActive] = useState(
    () => !!assetControls?.anyActive,
  );
  const pollAssets = !!assetControls && (assetPollTrigger || assetPollActive);
  const { assetState, translateState } = useEnrichmentPolling({
    runId,
    pollAssets,
    pollTranslate: translatePhase === "started",
  });

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
  // ~1/min, well within the default 1800s TTL).  An immediate catch-up
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

  // Release on unload (best-effort — sendBeacon for reliability) so another
  // operator can claim sooner.  A single operator has nobody to hand off to
  // and can always re-claim their own run, so the claim lapses by TTL
  // instead: releasing would race a reload or tutorial step that carries the
  // same token in its URL and leave the new page on a dead claim (#722).
  useEffect(() => {
    if (!claimed || !multiUser) return;
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
  }, [claimed, multiUser, runId, mediaId]);

  const editTextRef = useRef(editText);
  editTextRef.current = editText;
  const confirmDiscardRef = useRef(confirmDiscard);
  confirmDiscardRef.current = confirmDiscard;
  const claimLostRef = useRef(claimLost);
  claimLostRef.current = claimLost;
  const currentRef = useRef<Segment | null>(null);

  // The box holds text the line does not have yet. A save still in flight
  // counts: it can fail, and the box is the only copy until it lands.
  const hasUnsavedEdit = useCallback((): boolean => {
    const cur = currentRef.current;
    return cur !== null && editTextRef.current !== (cur.text ?? "");
  }, []);

  // Asked before the cursor leaves its line (issue #732). An unsaved edit gets
  // the same warn-then-continue as verify and split: the first move warns and a
  // repeated one discards. With the claim lost the edit cannot be saved and the
  // warning's panel is hidden, so the cursor stays put until a re-claim.
  const canLeave = useCallback((): boolean => {
    if (!hasUnsavedEdit()) return true;
    if (claimLostRef.current) return false;
    // Both the ref and the state, so two quick keys cannot both warn, and
    // the render that moves the cursor does not copy a stale `true` back into
    // the ref the keyboard focus effect reads.
    if (!confirmDiscardRef.current) {
      confirmDiscardRef.current = true;
      setConfirmDiscard(true);
      return false;
    }
    confirmDiscardRef.current = false;
    setConfirmDiscard(false);
    return true;
  }, [hasUnsavedEdit]);

  const playerRef = useRef<TranscriptPlayerHandle>(null);
  // A roster speaker's voice from another recording (issue #714). The sample
  // has its own audio element, so playback is made exclusive by hand: a sample
  // pauses the main player, and the main player starting stops the sample. It
  // lives for the editor's life, so closing the menu leaves a clip playing.
  const samplerRef = useRef<VoiceSamplePlayer | null>(null);
  const availabilityRef = useRef<AbortController | null>(null);
  useEffect(() => {
    samplerRef.current = createVoiceSamplePlayer(() =>
      playerRef.current?.pausePlayback(),
    );
    return () => {
      samplerRef.current?.dispose();
      availabilityRef.current?.abort();
      availabilityRef.current = null;
    };
  }, []);
  const onMainPlay = useCallback(() => samplerRef.current?.stop(), []);
  // Hearing is read-only: it never moves the cursor or touches the edit box,
  // so an unsaved edit is safe (issue #732). A failure is reported in the
  // editor, not the menu, which may have closed by the time it is known.
  const hearOtherRecording = useCallback(
    async (speakerId: string) => {
      const name =
        speakers.find((speaker) => speaker.id === speakerId)?.displayName ??
        "this speaker";
      setVoiceNotice(null);
      const outcome = await samplerRef.current?.play(
        `/media/${mediaId}/editor/voice-sample/${speakerId}`,
      );
      if (outcome === "none") {
        setVoiceNotice(
          `No confirmed line of ${name} from another recording is available.`,
        );
      } else if (outcome === "gone") {
        setVoiceNotice(
          `The recordings with ${name}'s confirmed lines can't be played anymore.`,
        );
      } else if (outcome === "failed") {
        setVoiceNotice(`Couldn't play ${name}'s voice. Try again.`);
      }
    },
    [mediaId, speakers],
  );
  const editRef = useRef<HTMLTextAreaElement>(null);
  const annotationRootRef = useRef<HTMLDivElement>(null);
  const reloadAnnotationsRef = useRef<(() => Promise<void>) | null>(null);

  const play = useCallback(
    (index: number) => playerRef.current?.playSegment(index),
    [],
  );

  const writable = reviewToken !== null && !claimLost;
  const [undoInfo, setUndoInfo] = useState<LabelsResult["undo"] | null>(null);
  const [mergeSuggestion, setMergeSuggestion] = useState<MergeSuggestion | null>(null);
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
  // Who has a voice sample in another recording (issue #714). Asked on the
  // first menu open and kept for the editor's life: eligibility depends only
  // on other recordings, so nothing done here can change it.
  const [sampleSpeakerIds, setSampleSpeakerIds] = useState<string[] | null>(null);
  useEffect(() => {
    if (!popoverTarget || sampleSpeakerIds !== null || availabilityRef.current) return;
    const controller = new AbortController();
    availabilityRef.current = controller;
    void (async () => {
      try {
        const response = await apiFetch(`/media/${mediaId}/editor/voice-samples`, {
          headers: { accept: "application/json" },
          cache: "no-store",
          signal: controller.signal,
        });
        const body = (await response.json()) as { speakerIds?: unknown };
        if (controller.signal.aborted || !Array.isArray(body.speakerIds)) return;
        setSampleSpeakerIds(body.speakerIds as string[]);
      } catch {
        // A later menu open may retry; hearing never blocks editing.
      } finally {
        if (availabilityRef.current === controller) availabilityRef.current = null;
      }
    })();
  }, [popoverTarget, sampleSpeakerIds, mediaId]);
  const popoverOpenRef = useRef(false);
  useEffect(() => {
    popoverOpenRef.current = popoverTarget != null;
  }, [popoverTarget]);
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

  const { cursor, setCursor, goTo, select, jumpNext, remaining } = useWalkCursor(
    segments,
    initialSegments,
    play,
    canLeave,
  );
  useEffect(() => {
    if (pendingOpenRef.current) {
      pendingOpenRef.current = false;
      return;
    }
    setPopoverTarget(null);
  }, [cursor, writable]);
  const [pendingPopoverIndex, setPendingPopoverIndex] = useState<number | null>(null);
  useLayoutEffect(() => {
    if (pendingPopoverIndex == null) return;
    if (pendingPopoverIndex !== cursor) {
      setPendingPopoverIndex(null);
      return;
    }
    setPendingPopoverIndex(null);
    const row = playerRef.current?.focusCursorRow();
    const btn = row?.querySelector<HTMLElement>(".tp-speaker-btn");
    if (btn) {
      btn.focus();
      pendingOpenRef.current = true;
      handleSpeakerClick(pendingPopoverIndex, btn.getBoundingClientRect());
    }
  }, [cursor, pendingPopoverIndex, handleSpeakerClick]);

  // A jump to a place in the recording (outline, annotation, the rail's hear
  // voice) plays it even when the cursor has to stay with an unsaved edit, as a
  // transcript click does; only the move waits.
  const jumpTo = useCallback(
    (index: number) => {
      if (!goTo(index) && index >= 0) play(index);
    },
    [goTo, play],
  );

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
      if (index >= 0) jumpTo(index);
    },
    [segments, jumpTo],
  );
  const hearPopoverVoice = useCallback(() => {
    if (popoverTarget) {
      playerRef.current?.previewSegment(popoverTarget.segmentIndex);
    }
  }, [popoverTarget]);

  const writeGuard = useMemo(
    () => ({ busy, busyRef, setBusy }),
    [busy, busyRef, setBusy],
  );

  // Roster voices to compare against in the popover (issue #571), each with one
  // line to preview. Only lines under a different diarization label than the
  // popover's count: those are a different voice cluster, so the speaker the
  // line is currently attributed to can be checked against their other speech.
  // Identity comes from the label's resolution or the child's own word-range
  // override; a line showing another roster name than its resolved label is a
  // whole-segment override and maps by that (unique) name. Lines of unresolved
  // labels never map by name, so a "Voice N" placeholder cannot pose as a
  // roster speaker. Per speaker, an operator-attributed line of at least two
  // seconds wins, then any operator line, then any line; ties keep the earliest.
  const comparisons = useMemo(() => {
    if (!popoverSegment) return [];
    const idByName = new Map(speakers.map((speaker) => [speaker.displayName, speaker.id]));
    const humanLabels = new Set(
      labelStates.filter((ls) => ls.resolution === "human_assign").map((ls) => ls.label),
    );
    const best = new Map<string, { index: number; score: number }>();
    segments.forEach((segment, index) => {
      if (segment.label === popoverSegment.label) return;
      const resolution =
        segment.label != null ? labelResolutions.get(segment.label) : undefined;
      let speakerId: string | null | undefined = null;
      let byOperator = true;
      if (segment.wordRangeSpeakerId != null) {
        speakerId = segment.wordRangeSpeakerId;
      } else if (resolution?.speakerId != null) {
        if (segment.speaker === resolution.speakerName) {
          speakerId = resolution.speakerId;
          byOperator = segment.label != null && humanLabels.has(segment.label);
        } else {
          speakerId = idByName.get(segment.speaker);
        }
      }
      if (speakerId == null) return;
      const score = (byOperator ? 2 : 0) + (segment.end - segment.start >= 2 ? 1 : 0);
      const previous = best.get(speakerId);
      if (previous === undefined || score > previous.score) {
        best.set(speakerId, { index, score });
      }
    });
    return speakers.flatMap((speaker) => {
      const found = best.get(speaker.id);
      return found ? [{ speaker, index: found.index }] : [];
    });
  }, [popoverSegment, speakers, labelStates, labelResolutions, segments]);
  const comparableSpeakers = useMemo(
    () => comparisons.map((comparison) => comparison.speaker),
    [comparisons],
  );
  // Everyone with a sample elsewhere, minus those the in-recording list above
  // already offers. The line's own speaker stays, so a doubtful match can be
  // checked. With seek disabled that list is not shown, so nobody is dropped.
  const otherRecordingSpeakers = useMemo(() => {
    const available = new Set(sampleSpeakerIds);
    const offered = new Set(
      capability.seekEnabled ? comparableSpeakers.map((speaker) => speaker.id) : [],
    );
    return speakers
      .filter((speaker) => available.has(speaker.id) && !offered.has(speaker.id))
      .sort((a, b) => a.displayName.localeCompare(b.displayName));
  }, [sampleSpeakerIds, speakers, comparableSpeakers, capability.seekEnabled]);
  const hearSpeaker = useCallback(
    (speakerId: string) => {
      const found = comparisons.find((comparison) => comparison.speaker.id === speakerId);
      if (found) playerRef.current?.previewSegment(found.index);
    },
    [comparisons],
  );

  const postForm = useFormPost(
    reviewToken,
    writable,
    () => setClaimLost(true),
    setError,
  );
  const applyResult = useSegmentPatch(segments, setSegments, setProgress);

  // Counts whole-run adoptions, so a refetch can tell that a newer result landed
  // while it was in flight. Every writer now shares the editor's write guard,
  // which the refetch runs under, so this is a backstop (issue #726).
  const labelsAdoptedRef = useRef(0);
  // The label states the next adoption diffs against. Updated on adoption, not
  // only on render, so two results adopted before a re-render still compare
  // each against the one before it (issue #726).
  const labelStatesRef = useRef(labelStates);
  useEffect(() => {
    labelStatesRef.current = labelStates;
  }, [labelStates]);

  // `keepUndo` leaves the undo toast up: a refetch after a refused undo must
  // not clear the toast that explains the refusal.
  const onLabelsChanged = useCallback(
    (result: LabelsResult, { keepUndo = false }: { keepUndo?: boolean } = {}) => {
      labelsAdoptedRef.current += 1;
      setMergeSuggestion(null);
      const changedLabels = new Set<string>();
      for (const newLs of result.labels) {
        const oldLs = labelStatesRef.current.find((ls) => ls.label === newLs.label);
        if (!oldLs || oldLs.speakerId !== newLs.speakerId || oldLs.resolution !== newLs.resolution) {
          changedLabels.add(newLs.label);
        }
      }
      if (changedLabels.size > 0) setHighlightLabels(changedLabels);

      setSegments(result.segments);
      setProgress(result.progress);
      labelStatesRef.current = result.labels;
      setLabelStates(result.labels);
      if (!keepUndo) setUndoInfo(result.undo ?? null);
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
      void refreshMarksRef.current();
    },
    [setSegments, setProgress],
  );

  // After a refused undo (issue #718) the scope was re-ruled elsewhere, so this
  // editor is stale. Adopt the whole run from the server. The undo toast holds
  // the editor write guard until this settles, so the request is time-boxed. On
  // failure the editor stays as it was until the next write or reload; the
  // toast already says the undo did not apply. Never rejects.
  const refetchAfterUndoConflict = useCallback(async () => {
    const adoptedBefore = labelsAdoptedRef.current;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UNDO_REFETCH_TIMEOUT_MS);
    try {
      const res = await apiFetch(`/review/${runId}/labels`, {
        headers: { accept: "application/json" },
        cache: "no-store",
        signal: controller.signal,
      });
      const result = (await res.json()) as LabelsResult;
      // A newer result is already on screen; this snapshot may predate it.
      if (labelsAdoptedRef.current !== adoptedBefore) return;
      // The refused action's "Assigned to …" announcement no longer holds.
      setAssignStatus(null);
      onLabelsChanged(result, { keepUndo: true });
    } catch (err) {
      console.warn("Could not refresh the editor after a refused undo", err);
    } finally {
      clearTimeout(timer);
    }
  }, [runId, onLabelsChanged]);

  const current =
    cursor >= 0 && cursor < segments.length ? segments[cursor] : null;
  currentRef.current = current;
  const focusParentId = current?.sourceSegmentId ?? null;
  const desiredMarksFocus = cleanupMode && writable ? focusParentId : null;
  useLayoutEffect(() => {
    // Record the latest committed focus before a POST can settle. The hook
    // owns deferral and settlement; busy state is only for rendering controls.
    requestMarksFocus(desiredMarksFocus);
  }, [desiredMarksFocus, requestMarksFocus]);
  useEffect(() => {
    setCleanupNotice(null);
  }, [current?.segmentId]);
  const marksByLine = useMemo(() => marksSnapshot
    ? wordMarksByLine(segments, marksSnapshot.payload) : undefined, [segments, marksSnapshot]);
  const staleByLine = useMemo(() => marksSnapshot
    ? staleMarksByLine(segments, marksSnapshot.payload) : undefined, [segments, marksSnapshot]);
  const cleanupEmission = marksSnapshot?.focus === focusParentId ? marksByLine?.get(cursor) : undefined;
  const cleanupLine = emissionKey(focusParentId, current?.wordStart ?? null);
  const focusedUnit = selectedUnit?.line === cleanupLine
    ? cleanupEmission?.units.find((u) => u.start === selectedUnit.start && u.end === selectedUnit.end) : undefined;
  const focusedUnitRef = useRef(focusedUnit);
  focusedUnitRef.current = focusedUnit;
  const onFocusUnit = useCallback((unit: WordMarkUnit) => {
    // A focus event and a shortcut can arrive before the next render.
    focusedUnitRef.current = unit;
    setSelectedUnit({ line: cleanupLine, start: unit.start, end: unit.end });
  }, [cleanupLine]);
  const toggleCleanup = useCallback(() => {
    if (!writable) return;
    setCleanupMode((on) => !on);
    setSplitMode(false);
  }, [writable]);
  const writeWordMark = useCallback(async (segmentId: string, start: number, end: number, action: WordMarkAction) => {
    if (!writable || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    const request = beginMarksWrite();
    try {
      const result = await postForm<WordMarksResult>(`/review/${runId}/segments/${segmentId}/word-marks`, {
        nonce: makeNonce(), action, start: String(start), end: String(end),
      });
      if (!result) return;
      adoptMarks(result, segmentId, request);
      if (result.undo) {
        markUndoFocus.current = segmentId;
        setUndoInfo(result.undo);
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
      finishMarksWrite();
    }
  }, [writable, busyRef, setBusy, beginMarksWrite, finishMarksWrite, adoptMarks, postForm, runId]);
  const actOnUnit = useCallback((key: "f" | "o" | "clear") => {
    const unit = focusedUnitRef.current;
    if (!unit || focusParentId === null) return;
    if (key === "clear") {
      if (unit.mark) void writeWordMark(focusParentId, unit.start, unit.end, "clear");
      return;
    }
    const choice = toggleWordMark(unit, key, marksSnapshot?.payload.detectionError ?? null);
    if (choice.error) setError(choice.error);
    else if (choice.action) void writeWordMark(focusParentId, unit.start, unit.end, choice.action);
  }, [focusParentId, writeWordMark, marksSnapshot]);
  const isSplitParent = siblingCount(segments, focusParentId) > 1;
  const speakerDisplayName =
    current?.speaker?.trim() || current?.label?.trim() || "Unknown speaker";
  const rawLabel = current?.label?.trim() ?? "";
  const showRawLabel = rawLabel !== "" && rawLabel !== speakerDisplayName;

  // What the edit box last loaded, so a change to the current line's text
  // (a whole-run result adopted from the rail, an undo or a refetch) can tell
  // an unsaved edit from an untouched box (issue #726).
  const loadedRef = useRef<{ segmentId: string | null; text: string }>({
    segmentId: null,
    text: "",
  });
  useEffect(() => {
    const segmentId = current?.segmentId ?? null;
    const text = current?.text ?? "";
    const loaded = loadedRef.current;
    loadedRef.current = { segmentId, text };
    const edit = editTextRef.current;
    const sameLine = segmentId !== null && segmentId === loaded.segmentId;
    // saveEdit records the saved text first, so the box's own save lands here
    // as no change and never reads as an edit overtaken by new text.
    const ownSave = sameLine && text === loaded.text;
    if (sameLine && !ownSave && edit !== loaded.text && edit !== text) {
      // Same line, new text, unsaved edit: keep the edit. Verify still warns
      // before discarding it, and saving replaces the new text with it.
      setEditOvertaken(true);
      return;
    }
    if (!ownSave) setEditText(text);
    setEditOvertaken(false);
    setConfirmDiscard(false);
    setAssignStatus(null);
    setVoiceNotice(null);
    setMergeSuggestion(null);
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
      const segmentId = current.segmentId;
      const edit = editText;
      const result = await postJson(
        `/review/${runId}/segments/${segmentId}/verify`,
        { verified: "true" },
      );
      if (!result) return;
      const patched = applyResult(index, result);
      setConfirmDiscard(false);
      if (walkMode) {
        // Advance only from the verified line with the box as it was: a move
        // or typing during the verify belongs to the operator, and a warning
        // another move raised meanwhile is not consent for this advance.
        const next = nextTarget(patched, index + 1);
        const unchanged =
          currentRef.current?.segmentId === segmentId &&
          editTextRef.current === edit;
        if (next >= 0 && unchanged && goTo(next)) {
          keyboardNavRef.current = true;
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
      const segmentId = current.segmentId;
      const submitted = editText;
      const result = await postJson(
        `/review/${runId}/segments/${segmentId}/text`,
        { text: submitted },
      );
      if (!result) return;
      // Navigation stays live during a save, so touch the box only if it still
      // shows this line. Show what was saved unless the operator typed on: the
      // server can return other text than was typed (an empty box reverts to
      // the pipeline text).
      if (currentRef.current?.segmentId === segmentId) {
        loadedRef.current = { segmentId, text: result.text };
        if (editTextRef.current === submitted) setEditText(result.text);
        // A save that leaves the line's text as it was never reruns the sync
        // effect, so clear the overtaken note here.
        setEditOvertaken(false);
        setConfirmDiscard(false);
        editRef.current?.blur();
        setCleanupNotice(marksClearedNotice(result.marksCleared ?? 0));
      }
      applyResult(index, result, { supersedeProvenance: true });
      void reloadAnnotationsRef.current?.();
      void refreshMarksRef.current();
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [current, cursor, postJson, runId, editText, applyResult, segments, busyRef, setBusy]);

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
        );
        if (!result) return;
        setConfirmDiscard(false);
        setSegments(result.segments);
        setProgress(result.progress);
        void reloadAnnotationsRef.current?.();
        void refreshMarksRef.current();
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
        const result = await postForm<RelabelResult>(
          `/review/${runId}/segments/${seg.sourceSegmentId}/relabel`,
          body,
        );
        if (!result) return;
        setSegments(result.segments);
        setProgress(result.progress);
        // A replay carries no undo; keep any toast that is still valid.
        if (result.undo) setUndoInfo(result.undo);
        void reloadAnnotationsRef.current?.();
        void refreshMarksRef.current();
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
        const result = await postForm<RelabelResult>(
          `/review/${runId}/segments/${targetParentId}/relabel`,
          body,
        );
        if (!result) return;
        setSegments(result.segments);
        setProgress(result.progress);
        // A replay carries no undo; keep any toast that is still valid.
        if (result.undo) setUndoInfo(result.undo);
        void reloadAnnotationsRef.current?.();
        void refreshMarksRef.current();
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
      // The relabel names its segment, so an unsaved edit can stay on its line.
      if (!hasUnsavedEdit()) setCursor(popoverTarget.segmentIndex);
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
      );
      if (result) {
        onLabelsChanged(result);
        const name = speakers.find((speaker) => speaker.id === speakerId)?.displayName;
        const count = popoverResolution?.segmentCount ?? 0;
        setAssignStatus(`Assigned ${count} ${count === 1 ? "segment" : "segments"} to ${name ?? "speaker"}.`);
        const suggestion = findMergeCandidates(result.labels, popoverSegment.label, speakerId);
        setMergeSuggestion(suggestion);
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [popoverTarget, popoverSegment, writable, busyRef, closePopover, hasUnsavedEdit, setCursor, reassignSegment, reassignChild, setBusy, postForm, runId, onLabelsChanged, speakers, popoverResolution]);

  const handleRailAssignment = useCallback(
    (label: string, speakerId: string, freshLabels: LabelStateShape[]) => {
      const suggestion = findMergeCandidates(freshLabels, label, speakerId);
      setMergeSuggestion(suggestion);
    },
    [],
  );

  const dismissMergeSuggestion = useCallback(() => setMergeSuggestion(null), []);
  const handleMergeSuggestionMerged = useCallback(
    (data: LabelsResult) => {
      setMergeSuggestion(null);
      onLabelsChanged(data);
    },
    [onLabelsChanged],
  );

  const handlePopoverReset = useCallback(async (scope: "segment" | "label") => {
    if (!popoverTarget || !popoverSegment || !writable || busyRef.current) return;
    closePopover();
    if (scope === "label") {
      setError("Reset is only supported for just this segment. Choose that scope to reset.");
      return;
    }
    if (!hasUnsavedEdit()) setCursor(popoverTarget.segmentIndex);
    const isChild = popoverSegment.wordStart != null && popoverSegment.wordEnd != null;
    if (isChild) {
      await reassignChild(popoverSegment, null);
    } else {
      await reassignSegment(null, popoverSegment);
    }
  }, [popoverTarget, popoverSegment, writable, busyRef, closePopover, hasUnsavedEdit, setCursor, reassignSegment, reassignChild]);

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
      setSegments((prev) => prev.map((seg) =>
        seg.speaker === popoverSegment.speaker
          ? { ...seg, speaker: data.displayName } : seg));
      setLabelStates((prev) => prev.map((ls) => ({
        ...ls,
        speakerName: ls.speakerId === data.id ? data.displayName : ls.speakerName,
        cosineSpeakerName: ls.cosineSpeakerId === data.id ? data.displayName : ls.cosineSpeakerName,
        candidateSpeakerName: ls.candidateSpeakerId === data.id ? data.displayName : ls.candidateSpeakerName,
      })));
      setAssignStatus(`Renamed speaker to ${data.displayName}.`);
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
    if (!translate || !translateTarget) return;
    translateBusyRef.current = true;
    setTranslateError(null);
    setTranslatePhase("starting");
    try {
      const body = new URLSearchParams({
        csrf_token: translate.csrf,
        target_language: translateTarget,
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
  }, [translate, translateTarget, runId]);

  const translateJobId = translateState
    ? translateState.activeJobId
    : (translate?.activeJobId ?? null);

  const cancelTranslation = useCallback(async () => {
    if (translateBusyRef.current) return;
    if (!translateJobId || !translate?.csrfCancel) return;
    translateBusyRef.current = true;
    setTranslateError(null);
    try {
      const body = new URLSearchParams({ csrf_token: translate.csrfCancel });
      const res = await apiFetch(
        `/runs/${runId}/translation/${translateJobId}/cancel`,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body,
        },
      );
      const data = (await res.json()) as {
        cancelled: boolean;
        error?: string | null;
      };
      if (data.cancelled) {
        setTranslatePhase("idle");
      } else {
        setTranslateError(data.error ?? "Translation could not be cancelled.");
        setTranslatePhase("error");
      }
    } catch (err) {
      setTranslateError(
        err instanceof ApiError ? err.detail : "Translation could not be cancelled.",
      );
      setTranslatePhase("error");
    } finally {
      translateBusyRef.current = false;
    }
  }, [translate, translateJobId, runId]);

  useEffect(() => {
    if (translatePhase === "started" && translateState && !translateState.active) {
      const failed =
        translateState.jobStatus === "FAILED" ||
        translateState.jobStatus === "CANCELLED";
      if (failed) {
        setTranslateError("Translation failed. Reload for details, then retry.");
        setTranslatePhase("error");
      } else {
        setTranslatePhase("idle");
      }
    }
  }, [translatePhase, translateState]);

  useEffect(() => {
    if (assetState && !assetState.anyActive) {
      if (assetPollTrigger) setAssetPollTrigger(false);
      if (assetPollActive) setAssetPollActive(false);
    }
  }, [assetState, assetPollTrigger, assetPollActive]);

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
    onJump: jumpTo,
    onClaimLost: onAnnotationClaimLost,
  });
  useEffect(() => {
    reloadAnnotationsRef.current = reloadAnnotations;
  }, [reloadAnnotations]);

  // Global keymap — same pattern as ReviewStepper, plus `w` for walk-mode toggle.
  useEffect(() => {
    if (!writable) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
      // A held key must not answer its own unsaved-edit warning: discarding
      // takes a second, deliberate press.
      if (event.repeat && hasUnsavedEdit()) return;
      if (helpOpenRef.current) return;
      if (popoverOpenRef.current) return;
      const el = event.target as HTMLElement | null;
      const tag = el?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el?.isContentEditable
      )
        return;
      // Word buttons are intentionally not form controls here: navigation and
      // verify stay available, and a cursor move fetches its new parent's units.
      switch (event.key.toLowerCase()) {
        case REVIEW_KEY.cleanup:
          event.preventDefault();
          toggleCleanup();
          break;
        case REVIEW_KEY.keepFiller:
        case REVIEW_KEY.omitWord:
          if (cleanupMode) {
            event.preventDefault();
            actOnUnit(event.key.toLowerCase() as "f" | "o");
          }
          break;
        case "escape":
          if (cleanupMode) {
            event.preventDefault();
            setCleanupMode(false);
            playerRef.current?.focusCursorRow();
          }
          break;
        case REVIEW_KEY.verify:
          event.preventDefault();
          void verifyAndAdvance();
          break;
        case REVIEW_KEY.skip:
          event.preventDefault();
          // Arm the focus only for a move that went; jumpNext can land on
          // the current line, which never runs the focus effect to disarm it.
          if (jumpNext()) keyboardNavRef.current = true;
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
          if (next !== cursor && goTo(next)) keyboardNavRef.current = true;
          break;
        }
        case REVIEW_KEY.previous: {
          event.preventDefault();
          const prev = Math.max(cursor - 1, 0);
          if (prev !== cursor && goTo(prev)) keyboardNavRef.current = true;
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
          if (cursor <= 0) {
            setAssignStatus("No previous segment to copy from.");
            break;
          }
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
    cleanupMode,
    toggleCleanup,
    actOnUnit,
    hasUnsavedEdit,
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
  // Shown with the lost-claim notice, since the edit box needs the claim.
  const lostClaimEdit =
    claimLost && current !== null && editText !== (current.text ?? "");

  return (
    <>
      <div
        ref={annotationRootRef}
        className={writable && walkMode ? "walk-active" : undefined}
      >
        {claimLost && (
          <div role="alert" className="notice text-sm">
            <p>
              Your claim expired or was taken over. Everything you already saved
              is safe.{" "}
              {lostClaimEdit
                ? "Your unsaved edit is below, and the editor stays on its line until you re-claim. Copy it, or "
                : ""}
              <button
                type="button"
                onClick={() => void claimForEditing()}
                disabled={claiming}
                className="underline"
              >
                {claiming
                  ? "Re-claiming…"
                  : lostClaimEdit
                    ? "re-claim to keep editing it"
                    : "Re-claim to continue editing"}
              </button>
              .
            </p>
            {lostClaimEdit && (
              // Read-only, not disabled, so the text can still be selected
              // and copied (issue #734).
              <textarea
                readOnly
                value={editText}
                rows={2}
                className="w-full text-sm"
                aria-label="Your unsaved edit"
              />
            )}
          </div>
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
                onClick={() => { setSplitMode((on) => !on); setCleanupMode(false); }}
                aria-pressed={splitMode}
                className="text-sm"
              >
                {splitMode ? "Exit split mode" : "Split"}
              </button>
              <button type="button" onClick={toggleCleanup} aria-pressed={cleanupMode} className="text-sm">
                {cleanupMode ? "Exit clean-up" : "Clean up"} <kbd>{REVIEW_KEY.cleanup}</kbd>
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
          {cleanupNotice && <p role="status" className="text-sm">{cleanupNotice}</p>}
          {marksSnapshot?.payload.detectionError && <p role="note" className="text-sm">
            Filler detection is unavailable: {marksSnapshot.payload.detectionError} You can still omit words, clear marks and undo.
          </p>}
          {cleanupMode && writable && <div className="me-actions text-sm" role="region" aria-label="Clean-up mode">
            <span>Clean-up mode: <kbd>{REVIEW_KEY.keepFiller}</kbd> keep, <kbd>{REVIEW_KEY.omitWord}</kbd> omit,
              {" "}<kbd>←</kbd> / <kbd>→</kbd> or <kbd>Tab</kbd> focus a word, <kbd>Escape</kbd> done.</span>
            {!cleanupEmission && <span role="status">{focusParentId === null
              ? "This line has no recorded segment to mark."
              : marksLoadError ? "Could not load words. Exit and re-enter clean-up mode to try again." : "Loading words…"}</span>}
            {cleanupEmission && !cleanupEmission.markable && <span role="status">{cleanupEmission.reason}</span>}
            <button type="button" disabled={busy || !focusedUnit} onClick={() => actOnUnit("f")}>Keep <kbd>{REVIEW_KEY.keepFiller}</kbd></button>
            <button type="button" disabled={busy || !focusedUnit} onClick={() => actOnUnit("o")}>Omit <kbd>{REVIEW_KEY.omitWord}</kbd></button>
            <button type="button" disabled={busy || !focusedUnit?.mark} onClick={() => actOnUnit("clear")}>Clear</button>
            <button type="button" onClick={() => setCleanupMode(false)}>Done <kbd>Escape</kbd></button>
          </div>}
          {translate &&
            (translatePhase === "idle" ? (
              <div className="me-actions">
                <select
                  aria-label="Translation language"
                  value={translateTarget}
                  onChange={(event) => setTranslateTarget(event.target.value)}
                  className="text-sm"
                >
                  <option value="" disabled>
                    Select a language
                  </option>
                  {translate.languageOptions.map(({ code, name }) => (
                    <option key={code} value={code}>
                      {name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => void startTranslate()}
                  disabled={!translateTarget}
                  className="text-sm"
                >
                  {translate.hasTranslation ? "Re-translate" : "Translate"}
                </button>
              </div>
            ) : translatePhase === "starting" ? (
              <span className="muted text-sm">Starting translation…</span>
            ) : translatePhase === "started" ? (
              <span className="muted text-sm">
                {translateState?.jobStatus
                  ? `Translating (${translateState.jobStatus})...`
                  : "Translation started."}{" "}
                The result appears on the{" "}
                <a href={translate.transcriptUrl}>transcript page</a>.
                {translateJobId && translate.csrfCancel && (
                  <>
                    {" "}
                    <button
                      type="button"
                      onClick={() => void cancelTranslation()}
                      className="text-sm"
                    >
                      Cancel
                    </button>
                  </>
                )}
              </span>
            ) : (
              <span role="alert" className="text-sm">
                {translateError}{" "}
                <button
                  type="button"
                  onClick={() => setTranslatePhase("idle")}
                  className="text-sm"
                >
                  Retry
                </button>{" "}
                <a href={`/runs/${runId}`}>Open the run page</a>
              </span>
            ))}
        </section>

        {writable && unresolvedCount > 0 && (
          <div
            className="me-unresolved-banner"
            role="region"
            aria-label="Unresolved voices"
            style={{
              padding: "0.5rem 0.75rem",
              background: "var(--surface-2)",
              color: "var(--ink-muted)",
              fontSize: "0.875rem",
            }}
          >
            <span aria-live="polite">
              {unresolvedCount} unidentified {unresolvedCount === 1 ? "voice" : "voices"},{" "}
              {affectedSegmentCount} {affectedSegmentCount === 1 ? "segment" : "segments"} affected
            </span>{" · "}
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                if (busyRef.current) return;
                const index = segments.findIndex(
                  (segment) => segment.label != null && needsYouLabels.has(segment.label),
                );
                if (index < 0) return;
                if (goTo(index)) setPendingPopoverIndex(index);
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
                {voiceNotice && (
                  <p role="status" className="text-sm">
                    {voiceNotice}
                  </p>
                )}
                {editOvertaken && (
                  <p role="status" className="text-sm">
                    This line changed while you were editing. Your edit is
                    kept; saving it replaces the new text.
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
            {/* The same notice when the panel above is not rendered. */}
            {!(writable && current && current.segmentId !== null) && voiceNotice && (
              <p role="status" className="text-sm">
                {voiceNotice}
              </p>
            )}
            <TranscriptPlayer
              ref={playerRef}
              onMainPlay={onMainPlay}
              runId={runId}
              mediaUrl={mediaUrl}
              segments={segments}
              capability={capability}
              lowConfidenceThreshold={lowConfidenceThreshold}
              onSegmentSelect={writable ? select : undefined}
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
              wordMarks={marksByLine}
              cleanupIndex={cleanupMode && writable && cleanupEmission ? cursor : null}
              onFocusUnit={onFocusUnit}
              staleWordMarks={staleByLine}
              wordMarksBusy={busy}
              onClearStaleMark={writable ? (mark) => void writeWordMark(mark.segmentId, mark.start, mark.end, "clear") : undefined}
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
                onHearVoice={capability.seekEnabled ? hearPopoverVoice : undefined}
                otherRecordingSpeakers={otherRecordingSpeakers}
                onHearOtherRecording={hearOtherRecording}
                comparableSpeakers={comparableSpeakers}
                onHearSpeaker={capability.seekEnabled ? hearSpeaker : undefined}
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
            onAssignment={handleRailAssignment}
            onHearVoice={capability.seekEnabled ? hearVoice : undefined}
            hearableLabels={hearableLabels}
            writeGuard={writeGuard}
          />
        </div>

        <OutlinePanel
          runId={runId}
          outline={outline}
          segments={segments}
          capability={capability}
          onJump={jumpTo}
          assetControls={
            assetControls
              ? {
                  ...assetControls,
                  polledState: assetState,
                  onActive: () => setAssetPollTrigger(true),
                }
              : undefined
          }
        />

        <KeymapHelp
          open={helpOpen}
          onClose={() => setHelpOpen(false)}
          hasRoster={speakers.length > 0}
          extraShortcuts={[
            { keys: REVIEW_KEY.cleanup, desc: "Toggle clean-up mode (leaves split mode)" },
            { keys: REVIEW_KEY.keepFiller, desc: "Toggle keep on the focused filler in clean-up mode" },
            { keys: REVIEW_KEY.omitWord, desc: "Toggle omit on the focused word in clean-up mode" },
            { keys: "Escape", desc: "Leave clean-up mode" },
            {
              keys: REVIEW_KEY.walkMode,
              desc: "Toggle walk mode (auto-advance after verify)",
            },
          ]}
        />
      </div>
      {undoInfo && writable && reviewToken && (
        <UndoToast
          key={undoInfo.kind === "merge" ? undoInfo.mergeNonce : undoInfo.kind === "word-mark" ? undoInfo.markId : undoInfo.decisionId}
          undo={undoInfo}
          runId={runId}
          reviewToken={reviewToken}
          claimCsrf={claimCsrf}
          onClaimLost={onAnnotationClaimLost}
          onUndone={(data) => {
            setUndoInfo(null);
            setAssignStatus(null);
            onLabelsChanged(data);
          }}
          onUndoStart={() => {
            if (undoInfo.kind === "word-mark") markUndoRequest.current = beginMarksWrite();
          }}
          onUndoSettled={() => {
            if (undoInfo.kind === "word-mark") finishMarksWrite();
          }}
          onMarksUndone={(data) => {
            adoptMarks(data, markUndoFocus.current, markUndoRequest.current);
            setUndoInfo(null);
          }}
          onDismiss={() => setUndoInfo(null)}
          onConflict={undoInfo.kind === "word-mark" ? async () => {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), UNDO_REFETCH_TIMEOUT_MS);
            try { await refreshMarks(undefined, controller.signal); }
            finally { clearTimeout(timer); }
          } : refetchAfterUndoConflict}
          writeGuard={writeGuard}
        />
      )}
      {mergeSuggestion && reviewToken && (
        <MergeSuggestionToast
          key={`${mergeSuggestion.assignedLabel}:${mergeSuggestion.targetSpeakerId}`}
          suggestion={mergeSuggestion}
          runId={runId}
          reviewToken={reviewToken}
          onClaimLost={onAnnotationClaimLost}
          onMerged={handleMergeSuggestionMerged}
          onDismiss={dismissMergeSuggestion}
          stacked={!!undoInfo}
          writeGuard={writeGuard}
        />
      )}
    </>
  );
}
