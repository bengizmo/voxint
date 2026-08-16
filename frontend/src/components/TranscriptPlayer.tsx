import { useEffect, useRef, useState } from "react";

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

// Audio-synced transcript with per-line playback (issue #49). Native <audio>
// plus the segment list, highlighting the currently-playing segment via the
// element's `timeupdate` event. Per-line ▶ / click-to-seek play just that ASR
// line's span; both are gated on the fail-closed capability contract (issue
// #55) and, when disabled, an honest banner explains why.
export function TranscriptPlayer({
  runId,
  mediaUrl,
  segments,
  capability,
}: TranscriptPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [activeIndex, setActiveIndex] = useState<number>(-1);
  const [rate, setRate] = useState<number>(() => getStoredRate());

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

  return (
    <div>
      <div className="flex items-center my-2">
        <SpeedControl rate={rate} onChange={onRateChange} />
      </div>
      <audio ref={audioRef} controls src={mediaUrl} className="w-full my-2" data-run-id={runId}>
        Your browser does not support the audio element.
      </audio>
      <CapabilityBanner capability={capability} />
      <div>
        {segments.map((seg, i) => (
          <p
            key={`${seg.start}-${i}`}
            className={
              i === activeIndex
                ? "my-1 text-sm rounded bg-sky-500/20 px-1"
                : "my-1 text-sm opacity-85"
            }
            aria-current={i === activeIndex ? "true" : undefined}
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
            <strong>{seg.speaker}:</strong> {seg.text}
          </p>
        ))}
      </div>
    </div>
  );
}
