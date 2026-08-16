import { useCallback, useEffect, useRef, useState } from "react";

import {
  cancelActiveTurn,
  getStoredRate,
  playTurn,
  setStoredRate,
  type PlaybackCapability,
} from "../lib/playback";
import { CapabilityBanner, SpeedControl } from "./PlaybackControls";

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
}

export interface TranscriptPlayerProps {
  runId: string;
  mediaUrl: string;
  segments: Segment[];
  capability: PlaybackCapability;
}

function formatTime(seconds: number): string {
  return seconds.toFixed(2);
}

// While a programmatic scrollIntoView is settling, real scroll events it emits
// must NOT be read as the operator taking manual control. This window (ms) is
// generous enough to cover the synchronous (non-smooth) scroll's follow-up
// events without swallowing a genuine user scroll a moment later.
const SCROLL_GUARD_MS = 200;

// Audio-synced transcript with per-line playback (issue #49) and follow-along
// highlight + per-speaker colors (issue #50). Native <audio> plus the segment
// list, highlighting the currently-playing segment via the element's
// `timeupdate` event. Per-line ▶ / click-to-seek play just that ASR line's
// span; both are gated on the fail-closed capability contract (issue #55) and,
// when disabled, an honest banner explains why. Color is SUPPLEMENTAL — the raw
// label badge is the primary, non-color identity cue (accessibility).
export function TranscriptPlayer({
  runId,
  mediaUrl,
  segments,
  capability,
}: TranscriptPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [activeIndex, setActiveIndex] = useState<number>(-1);
  const [rate, setRate] = useState<number>(() => getStoredRate());
  // Follow-along: keep the active line in view as playback advances. Starts on;
  // any manual scroll turns it off; the single "Resume following" control turns
  // it back on. No always-on checkbox, no status dot.
  const [following, setFollowing] = useState<boolean>(true);
  const activeLineRef = useRef<HTMLParagraphElement | null>(null);
  // Timestamp until which self-emitted scroll events are ignored (see guard).
  const scrollGuardUntil = useRef<number>(0);

  // Scroll the active line into view WITHOUT moving DOM focus (accessibility)
  // and WITHOUT smooth scrolling. Arm the guard first so the scroll events this
  // triggers are not misread as a manual scroll that would stop following.
  const scrollActiveIntoView = useCallback(() => {
    const el = activeLineRef.current;
    if (!el) return;
    scrollGuardUntil.current = performance.now() + SCROLL_GUARD_MS;
    el.scrollIntoView({ block: "nearest" });
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTimeUpdate = () => {
      const t = audio.currentTime;
      const idx = segments.findIndex((seg) => t >= seg.start && t < seg.end);
      setActiveIndex(idx);
    };
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      // On unmount, cancel any in-flight turn guard and stop the stream so
      // switching runs can't leave it open.
      cancelActiveTurn();
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    };
  }, [segments]);

  // Any real scroll (wheel, touch, keyboard, scrollbar) that is NOT one we just
  // triggered means the operator took over — stop following. Passive listener;
  // symmetric teardown keeps it StrictMode-safe.
  useEffect(() => {
    const onScroll = () => {
      if (performance.now() < scrollGuardUntil.current) return;
      setFollowing(false);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
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

  const seek = capability.seekEnabled;
  const play = (seg: Segment) => {
    const audio = audioRef.current;
    if (audio && seek) playTurn(audio, seg.start, seg.end);
  };
  const onRateChange = (next: number) => {
    setRate(next);
    setStoredRate(next);
  };
  const resumeFollowing = () => {
    setFollowing(true);
    scrollActiveIntoView();
  };

  return (
    <div>
      <div className="flex items-center my-2">
        <SpeedControl rate={rate} onChange={onRateChange} />
        {!following && (
          <button type="button" onClick={resumeFollowing} className="text-sm mr-2">
            Resume following
          </button>
        )}
      </div>
      <audio ref={audioRef} controls src={mediaUrl} className="w-full my-2" data-run-id={runId}>
        Your browser does not support the audio element.
      </audio>
      <CapabilityBanner capability={capability} />
      <div>
        {segments.map((seg, i) => {
          const active = i === activeIndex;
          const classes = ["tp-line", "my-1", "text-sm"];
          if (seg.paletteIndex != null) classes.push(`spk-${seg.paletteIndex}`);
          classes.push(active ? "rounded" : "opacity-85");
          if (active) classes.push("bg-sky-500/20", "px-1");
          return (
            <p
              key={`${seg.start}-${i}`}
              ref={active ? activeLineRef : undefined}
              className={classes.join(" ")}
              aria-current={active ? "true" : undefined}
              // Click-the-line-to-seek (issue #49). Only a hint when seeking is
              // disabled — the button carries the accessible affordance.
              onClick={seek ? () => { play(seg); } : undefined}
              style={seek ? { cursor: "pointer" } : undefined}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  play(seg);
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
              {seg.label != null && <span className="spk-badge">{seg.label}</span>}
              {seg.speaker !== seg.label && <strong>{seg.speaker}:</strong>} {seg.text}
            </p>
          );
        })}
      </div>
    </div>
  );
}
