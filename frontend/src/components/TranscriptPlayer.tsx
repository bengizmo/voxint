import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import {
  cancelActiveTurn,
  getStoredRate,
  playTurn,
  setStoredRate,
  type PlaybackCapability,
} from "../lib/playback";
import { fetchPeaks, type PeaksPayload, type Turn } from "../lib/peaks";
import { CapabilityBanner, SpeedControl } from "./PlaybackControls";
import { WaveformStrip } from "./WaveformStrip";

// Deterministic domain-pack correction provenance (issue #83). Mirrors the
// server's `resolve_segment_provenance` shapes EXACTLY — the keys are pinned by
// tests/contracts/test_correction_provenance_contract.py, so a drift here rots the
// console silently. One shown entry = one rule that materially fired on this
// segment: `pack`/`match`/`replace` are the resolved rule identity (null when the
// fired id is absent from the run's frozen snapshot — the entry stays VISIBLE as
// "unresolved rule <id>", never dropped); `span` addresses the persisted enhanced
// text in its `inputBase` coordinate space, NOT the operator-effective text.
export interface CorrectionEntry {
  id: string;
  from: string;
  to: string;
  span: [number, number] | null;
  pack: string | null;
  match: string | null;
  replace: string | null;
  resolved: boolean;
}

// The per-segment `corrections` field: null when no rule materially fired (or on a
// split child), an honest "unavailable" state when the row was recorded by a
// different corrector version than this console reads, else the shown provenance.
export type SegmentCorrections =
  | { status: "shown"; version: number; inputBase: string; entries: CorrectionEntry[] }
  | { status: "unavailable"; reason: string; recordedVersion: number | null };

// One row of the run-level "declared but never fired" reconciliation (issue #83).
// Mirrors the server's `run_reconciliation` entry shape (contract-pinned). status:
// `applied` (fired on >=1 segment's raw, appliedCount>0), `no_raw_match` (matched no
// segment's raw), or `growth_rejected` (would fire but the raw transformation
// overflowed the growth ceiling). appliedCount is 0 for the non-applied statuses.
export interface ReconciliationEntry {
  id: string;
  pack: string;
  match: string;
  replace: string;
  status: "applied" | "no_raw_match" | "growth_rejected";
  appliedCount: number;
}

export interface Segment {
  start: number;
  end: number;
  speaker: string;
  text: string;
  // Raw diarization label (issue #50): the primary, non-color identity cue,
  // shared with the JS-off fallback and the workbench card. May be null.
  label: string | null;
  // Curated palette index [0, PALETTE_SIZE) or null (label outside the run's
  // canonical universe). Drives the `spk-N` class → the same CSS accent the
  // server-rendered fallback uses, so island and fallback color identically.
  paletteIndex: number | null;
  // ASR confidence (exp(avg_logprob), a transformed likelihood — NOT a
  // calibrated probability). null when unknown; never flagged when null.
  confidence: number | null;
  // Per-segment review state (issues #53/#58). segmentId is the write target for
  // verify/correct; null for a synthetic/blank line that is never a review
  // target. verified drives the verify-and-advance loop; corrected keys the
  // "edited" badge. Both default false on an un-reviewed segment.
  segmentId: string | null;
  verified: boolean;
  corrected: boolean;
  // Split provenance (issue #59). sourceSegmentId is the IMMUTABLE PARENT id —
  // the verify/correct/split write target — identical to segmentId for an
  // unsplit line and SHARED across a split parent's derived child lines.
  // reviewTarget is true on exactly one line per parent (the N-of-M queue
  // entry), so the loop counts one target per parent, never per child.
  sourceSegmentId: string | null;
  reviewTarget: boolean;
  // Word-range coordinates of a split child (issue #59 slice 3): the exact
  // half-open [wordStart, wordEnd) this line covers within its parent. Set only
  // on a split parent's derived children — the coordinates the per-child reassign
  // picker posts to /relabel to scope a ruling to this child alone. Both null on
  // unsplit lines and synthetic/blank lines (no partitionable range).
  wordStart: number | null;
  wordEnd: number | null;
  // The child's OWN active word-range override speaker id (issue #59 slice 3), or
  // null when the child merely inherits the segment's whole-segment/label speaker.
  // The reassign picker binds its <select> to this, so the control shows a
  // child-scoped assignment only when one truly exists (never an inherited speaker
  // mislabeled as a child ruling), and "inherit" is selected exactly when null.
  wordRangeSpeakerId: string | null;
  // Deterministic domain-pack correction provenance (issue #83): which pack + rule
  // produced each edit on this segment, or an honest "unavailable" state; null when
  // no rule materially fired (keyed off the persisted trace, never a text diff) or
  // on a split child (spans are parent-scoped, never a child slice).
  corrections: SegmentCorrections | null;
  // The immutable raw ASR text for the WHOLE segment (issue #83), for the console's
  // compare / reset-to-raw affordance. null on split children (raw is a
  // whole-segment concern) and synthetic/blank export lines.
  rawText: string | null;
}

// One word token of a segment (issue #59), from the lazy `/words` endpoint. The
// server names the string field `word`; timings are seconds into the media.
export interface SplitWord {
  start: number;
  end: number;
  word: string;
}

// The focused segment's split affordance (issue #59): the line index whose words
// render as clickable split points, the parent id every cut writes to, and the
// word tokens. Present only while split mode is engaged on an unsplit, splittable
// segment; null otherwise (ordinary text rendering).
export interface SplitFocus {
  segmentIndex: number;
  sourceSegmentId: string;
  words: SplitWord[];
}

export interface TranscriptPlayerProps {
  runId: string;
  mediaUrl: string;
  segments: Segment[];
  capability: PlaybackCapability;
  // Triage cutoff (issue #53): a segment with confidence < this is flagged
  // "uncertain". Same server setting the JS-off fallback compares against, so
  // the island and fallback flag identically.
  lowConfidenceThreshold: number;
  // Optional selection callback (issue #53): fired with the segment index when a
  // line (or its ▶) is activated, so a driver like ReviewStepper can move its
  // edit cursor to the line the operator clicked — a correction then lands on the
  // segment being played, never the one under a stale cursor. Absent on the
  // read-only transcript page, which stays byte-identical (play only).
  onSegmentSelect?: (index: number) => void;
  // Waveform strip (issue #57). peaksUrl is server-owned truth: null/absent ⇒
  // no fetch, no strip, rendered output unchanged. turns are the diarization
  // regions the strip paints (an honest who-spoke-when map — see peaks.ts).
  peaksUrl?: string | null;
  turns?: Turn[];
  // Review-cursor position for the strip's underline marker (ReviewStepper's
  // `cursor`). Absent on the read-only surface.
  cursorIndex?: number;
  // Split mode (issue #59). When set, the focused segment's words render as
  // clickable split points instead of its text; a click cuts "before" that word.
  // Null/absent ⇒ ordinary text rendering (the read-only surface and every
  // non-split-mode render pass nothing). onSplitAt raises the chosen cut to the
  // driver (ReviewStepper), which POSTs it — the player never touches the DB.
  splitFocus?: SplitFocus | null;
  onSplitAt?: (sourceSegmentId: string, wordIndex: number) => void;
  // Per-child reassign (issue #59 slice 3). Present only on the claim-gated review
  // surface. When onReassign is set, a split parent's derived child lines (those
  // carrying wordStart/wordEnd — the ONLY lines the backend ranges) render a
  // speaker <select>: choosing a roster speaker assigns just that child's word
  // range, choosing "inherit" resets it to follow the label. The player raises the
  // choice; ReviewStepper POSTs /relabel — the player never touches the DB.
  // reassignSpeakers is the ACTIVE roster; reassignBusy disables the picker while a
  // write is in flight. Absent on the read-only surface (no picker rendered).
  reassignSpeakers?: { id: string; displayName: string }[];
  onReassign?: (seg: Segment, speakerId: string | null) => void;
  reassignBusy?: boolean;
}

// Imperative handle (issue #53): the ONLY review affordance the pure player
// exposes. The verify-and-advance loop (ReviewStepper) commands playback of one
// segment through this — the same code path as clicking a line — so highlight
// and follow-scroll come free and the read-only page stays byte-identical
// (it renders the player with no ref).
export interface TranscriptPlayerHandle {
  playSegment: (index: number) => void;
}

function formatTime(seconds: number): string {
  return seconds.toFixed(2);
}

// A programmatic scrollIntoView emits `scroll` events that must NOT be read as
// the operator taking over. `wheel`/`touchmove`/keyboard scrolling are detected
// by their OWN events (which a programmatic scroll never emits), so they stop
// following immediately and unambiguously. The `scroll` listener is only the
// catch-all for scrollbar drag and momentum, and it alone needs this short guard
// window (ms) — armed ONLY when an auto-scroll actually moves the page — so a
// self-emitted scroll event isn't misread. Time is never the sole discriminator.
const SCROLL_GUARD_MS = 200;

// Keyboard keys that scroll the document. Space is deliberately excluded: it is
// also play/pause on a focused media element, and a genuine Space-scroll still
// produces a `scroll` event the catch-all listener picks up.
const SCROLL_KEYS = new Set([
  "PageUp",
  "PageDown",
  "Home",
  "End",
  "ArrowUp",
  "ArrowDown",
]);

// Audio-synced transcript with per-line playback (issue #49) and follow-along
// highlight + per-speaker colors (issue #50). Native <audio> plus the segment
// list, highlighting the currently-playing segment via the element's
// `timeupdate` event. Per-line ▶ / click-to-seek play just that ASR line's
// span; both are gated on the fail-closed capability contract (issue #55) and,
// when disabled, an honest banner explains why. Color is SUPPLEMENTAL — the raw
// label badge is the primary, non-color identity cue (accessibility).
export const TranscriptPlayer = forwardRef<
  TranscriptPlayerHandle,
  TranscriptPlayerProps
>(function TranscriptPlayer(
  {
    runId,
    mediaUrl,
    segments,
    capability,
    lowConfidenceThreshold,
    onSegmentSelect,
    peaksUrl,
    turns,
    cursorIndex,
    splitFocus,
    onSplitAt,
    reassignSpeakers,
    onReassign,
    reassignBusy,
  },
  ref,
) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState<number>(-1);
  const [rate, setRate] = useState<number>(() => getStoredRate());
  // Waveform envelope (issue #57): null until (and unless) the fetch succeeds
  // and validates — the strip is pure enhancement, absent on any failure.
  const [peaks, setPeaks] = useState<PeaksPayload | null>(null);
  // Playhead position, quantized to 0.5s so timeupdate never re-renders more
  // than ~2Hz (at overview scale one strip pixel spans far more than that).
  // Starts at -1 ("never played") so the strip hides the playhead until the
  // first timeupdate — and DOES show it while segment 0 (start = 0s) plays.
  const [playheadTime, setPlayheadTime] = useState<number>(-1);
  // Follow-along: keep the active line in view as playback advances. Starts on;
  // any manual scroll turns it off; the single "Resume following" control turns
  // it back on. No always-on checkbox, no status dot.
  const [following, setFollowing] = useState<boolean>(true);
  const activeLineRef = useRef<HTMLParagraphElement | null>(null);
  // Timestamp until which self-emitted scroll events are ignored (see guard).
  const scrollGuardUntil = useRef<number>(0);

  // Scroll the active line into view WITHOUT moving DOM focus (accessibility)
  // and WITHOUT smooth scrolling. If the line is already fully visible we do
  // nothing AND do not arm the guard — otherwise the guard would needlessly
  // ignore a real operator scroll during a window in which we never moved.
  const scrollActiveIntoView = useCallback(() => {
    const el = activeLineRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const fullyVisible = r.top >= 0 && r.bottom <= window.innerHeight;
    if (fullyVisible) return;
    scrollGuardUntil.current = performance.now() + SCROLL_GUARD_MS;
    el.scrollIntoView({ block: "nearest" });
  }, []);

  // Follow-along highlight + playhead. Re-subscribes when `segments` identity
  // changes (a verify/correct patches the array), but MUST NOT tear down the
  // media element here: ReviewStepper re-renders with a new `segments` array on
  // every successful write, and stripping `src`/`load()` on that path would
  // kill playback for the rest of the session (React never re-sets an unchanged
  // `src` prop). Teardown lives in its own unmount-only effect below.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTimeUpdate = () => {
      const t = audio.currentTime;
      const idx = segments.findIndex((seg) => t >= seg.start && t < seg.end);
      setActiveIndex(idx);
      // Quantized: setState with an unchanged value skips the re-render, so
      // this costs nothing between half-second boundaries. Any timeupdate (even
      // at t=0, playing segment 0) lifts the playhead off its -1 "never played"
      // sentinel so it becomes visible.
      setPlayheadTime(Math.round(t * 2) / 2);
    };
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, [segments]);

  // Unmount-only teardown: cancel any in-flight turn guard and stop the stream
  // so switching runs (a real unmount) can't leave it open. Empty deps — this
  // must never fire on a `segments` change (see the effect above).
  useEffect(() => {
    const audio = audioRef.current;
    return () => {
      if (!audio) return;
      cancelActiveTurn();
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    };
  }, []);

  // The operator taking over stops following. `wheel`/`touchmove`/scroll-key
  // presses are unambiguous manual intent — a programmatic scrollIntoView never
  // emits them — so they stop immediately, no guard needed. The `scroll`
  // listener is the catch-all (scrollbar drag, momentum) and is the only one
  // that must ignore the events our own auto-scroll emits, via the short guard.
  // All passive; symmetric teardown keeps it StrictMode-safe.
  useEffect(() => {
    const stop = () => {
      setFollowing(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (SCROLL_KEYS.has(event.key)) setFollowing(false);
    };
    const onScroll = () => {
      if (performance.now() < scrollGuardUntil.current) return;
      setFollowing(false);
    };
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("wheel", stop);
      window.removeEventListener("touchmove", stop);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  // Keep the active line visible as playback advances. activeIndex changes only
  // at segment boundaries, so this is cheap.
  useEffect(() => {
    if (following && activeIndex >= 0) scrollActiveIntoView();
  }, [following, activeIndex, scrollActiveIntoView]);

  // Keep the element's rate in sync with the (persisted) control.
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) audio.playbackRate = rate;
  }, [rate]);

  // One aborted-on-unmount peaks fetch (issue #57). No peaksUrl ⇒ nothing —
  // the server sends null when the route would only 404/410. The Abort +
  // stale-check pair keeps StrictMode's double-mount and a mid-flight prop
  // change from applying a superseded response.
  useEffect(() => {
    setPeaks(null);
    if (!peaksUrl) return;
    const controller = new AbortController();
    void fetchPeaks(peaksUrl, controller.signal).then((payload) => {
      if (!controller.signal.aborted) setPeaks(payload);
    });
    return () => {
      controller.abort();
    };
  }, [peaksUrl]);

  const seek = capability.seekEnabled;
  const play = (seg: Segment) => {
    const audio = audioRef.current;
    if (audio && seek) playTurn(audio, seg.start, seg.end);
  };
  // Activating a line plays it AND (when a driver is attached) selects it, so the
  // edit cursor tracks what the operator clicked. Without the callback this is
  // exactly `play` — the read-only page is unchanged.
  const activateLine = (index: number, seg: Segment) => {
    play(seg);
    onSegmentSelect?.(index);
  };
  // A waveform-region click (issue #57): ALWAYS select + reveal the segment in
  // the list (selection is a reading act), and additionally play it only when
  // seeking is trusted — the strip itself never touches the audio element, so
  // the fail-closed gate stays structural.
  const onRegionActivate = (index: number) => {
    const seg = segments[index];
    if (!seg) return;
    onSegmentSelect?.(index);
    const el = listRef.current?.querySelector<HTMLElement>(
      `[data-seg-index="${index}"]`,
    );
    if (el) {
      scrollGuardUntil.current = performance.now() + SCROLL_GUARD_MS;
      el.scrollIntoView({ block: "nearest" });
    }
    play(seg);
  };
  const onRateChange = (next: number) => {
    setRate(next);
    setStoredRate(next);
  };
  const resumeFollowing = () => {
    setFollowing(true);
    scrollActiveIntoView();
  };

  // Expose only "play this segment" to the review loop. Bounds-guarded and
  // capability-gated (via `play`), so a bad index or disabled seek is a no-op,
  // never a throw.
  useImperativeHandle(
    ref,
    () => ({
      playSegment: (index: number) => {
        const seg = segments[index];
        if (seg) play(seg);
      },
    }),
    // `play` closes over audioRef + seek (both stable for a given render);
    // segments identity is what actually changes the mapping.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [segments, seek],
  );

  return (
    <div>
      {/* Player surface (issue #92): the native <audio>, speed control, waveform
          and capability banner framed as one styled panel. The native control is
          wrapped, never replaced (report §9.1) — a custom transport is #95. */}
      <div className="player-surface">
        <div className="flex items-center my-2">
          <SpeedControl rate={rate} onChange={onRateChange} />
          {!following && (
            <button
              type="button"
              onClick={resumeFollowing}
              className="text-sm mr-2"
            >
              Resume following
            </button>
          )}
        </div>
        <audio
          ref={audioRef}
          controls
          src={mediaUrl}
          className="w-full my-2"
          data-run-id={runId}
        >
          Your browser does not support the audio element.
        </audio>
        {peaks && (
          <WaveformStrip
            peaks={peaks}
            turns={turns ?? []}
            segments={segments}
            activeIndex={activeIndex}
            cursorIndex={cursorIndex}
            seekEnabled={seek}
            currentTime={playheadTime}
            onRegionActivate={onRegionActivate}
          />
        )}
        <CapabilityBanner capability={capability} />
      </div>
      <div ref={listRef}>
        {segments.map((seg, i) => {
          const active = i === activeIndex;
          // Split mode (issue #59): this focused, unsplit, splittable line shows
          // its words as clickable cut points instead of its plain text. Guarded
          // on a matching parent id so a stale fetch never paints the wrong line.
          const splitting =
            splitFocus != null &&
            onSplitAt != null &&
            i === splitFocus.segmentIndex &&
            seg.sourceSegmentId === splitFocus.sourceSegmentId;
          // Uncertain is a NON-background cue (a dashed underline + chip): the
          // active line owns the background tint, so the two never collide.
          const uncertain =
            seg.confidence != null && seg.confidence < lowConfidenceThreshold;
          const classes = ["tp-line", "my-1", "text-sm"];
          if (seg.paletteIndex != null) classes.push(`spk-${seg.paletteIndex}`);
          if (uncertain) classes.push("tp-uncertain");
          // Base .tp-line owns padding + radius (issue #92) so the active tint
          // and the hover surface align without per-state padding utilities.
          if (active) classes.push("bg-seg/20");
          else classes.push("opacity-85");
          return (
            <p
              // Keyed by parent + word-range (falling back to start time) so a
              // whole-run reconcile after a split/reassign re-associates each line
              // with its own DOM node — a split parent's children share a start
              // time, so a start-only key could otherwise remount the wrong <select>
              // and drop its focus. Index keeps unsplit lines with equal starts
              // distinct.
              key={`${seg.sourceSegmentId ?? "x"}-${seg.wordStart ?? "u"}-${seg.wordEnd ?? "u"}-${seg.start}-${i}`}
              ref={active ? activeLineRef : undefined}
              data-seg-index={i}
              className={classes.join(" ")}
              aria-current={active ? "true" : undefined}
              // Click-the-line-to-seek (issue #49). Only a hint when seeking is
              // disabled — the button carries the accessible affordance.
              onClick={
                seek
                  ? () => {
                      activateLine(i, seg);
                    }
                  : undefined
              }
              style={seek ? { cursor: "pointer" } : undefined}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  activateLine(i, seg);
                }}
                disabled={!seek}
                title={seek ? "Play this line" : capability.reasons[0]?.message}
                aria-label={`Play line at ${formatTime(seg.start)} seconds`}
                className="mr-2"
              >
                ▶
              </button>
              <span className="opacity-60 tabular-nums mr-2">
                [{formatTime(seg.start)}–{formatTime(seg.end)}]
              </span>
              {uncertain && (
                <span
                  className="tp-uncertain-chip"
                  title="Low ASR confidence — uncertain, not necessarily wrong"
                >
                  uncertain
                </span>
              )}
              {seg.label != null && (
                <span className="spk-badge">{seg.label}</span>
              )}
              {seg.speaker !== seg.label && <strong>{seg.speaker}:</strong>}{" "}
              {splitting ? (
                // A cut lands BEFORE the clicked word, so word 0 has no legal cut
                // (its index would be 0, and the backend requires 0 < index < n);
                // it renders inert. stopPropagation keeps the click off the line's
                // seek handler — splitting must not also scrub the audio.
                <span>
                  {splitFocus.words.map((w, wi) => (
                    <button
                      key={wi}
                      type="button"
                      disabled={wi === 0}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSplitAt(splitFocus.sourceSegmentId, wi);
                      }}
                      title={
                        wi === 0
                          ? "Cannot split before the first word"
                          : `Split before “${w.word.trim()}”`
                      }
                      className="tp-split-word px-0.5 rounded hover:bg-splitword/30 disabled:opacity-60 disabled:cursor-default"
                    >
                      {w.word}
                    </button>
                  ))}
                </span>
              ) : (
                seg.text
              )}
              {onReassign != null &&
                reassignSpeakers != null &&
                seg.wordStart != null &&
                seg.wordEnd != null && (
                  <label
                    className="tp-reassign ml-2 text-xs opacity-80"
                    // Keep the whole control — label text included — off the line's
                    // seek handler, not just the <select>.
                    onClick={(e) => e.stopPropagation()}
                  >
                    {" · "}speaker:{" "}
                    <select
                      // Bound to the child's OWN range-override id: "" (inherit) is
                      // selected exactly when the child has no child-scoped ruling,
                      // so an inherited speaker is never shown as a child assignment.
                      // Server truth, so a failed write (no reconcile) re-imposes the
                      // prior value on the next render rather than leaving the picked
                      // option showing. WRITE is by id regardless.
                      value={seg.wordRangeSpeakerId ?? ""}
                      disabled={reassignBusy}
                      aria-label={`Reassign speaker for “${seg.text}”`}
                      onChange={(e) => {
                        e.stopPropagation();
                        onReassign(seg, e.target.value === "" ? null : e.target.value);
                      }}
                    >
                      <option value="">↺ inherit (follow the segment)</option>
                      {reassignSpeakers.map((sp) => (
                        <option key={sp.id} value={sp.id}>
                          {sp.displayName}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
            </p>
          );
        })}
      </div>
    </div>
  );
});
