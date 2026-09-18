import { useEffect, useState, type RefObject } from "react";
import { formatTime } from "../lib/format";
import {
  cancelActiveTurn,
  RATE_STEPS,
  type PlaybackCapability,
} from "../lib/playback";

// Subscribe to the media element so segment playback and external pauses are
// reflected here too, without re-rendering the transcript on every timeupdate.
export function AudioTransport({
  audioRef,
  mediaUrl,
}: {
  audioRef: RefObject<HTMLAudioElement | null>;
  mediaUrl: string;
}) {
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const sync = () => {
      const isPlaying = !audio.paused && !audio.ended;
      setPlaying(isPlaying);
      if (isPlaying) setError("");
      setPosition(Number.isFinite(audio.currentTime) ? audio.currentTime : 0);
      setDuration(
        Number.isFinite(audio.duration) ? Math.max(0, audio.duration) : 0,
      );
    };
    const reset = () => {
      cancelActiveTurn();
      setPlaying(false);
      setPosition(0);
      setDuration(0);
      setError("");
    };
    const fail = () => {
      setPlaying(false);
      setError("Audio is unavailable or could not be loaded.");
    };
    const events = [
      "play",
      "pause",
      "ended",
      "timeupdate",
      "loadedmetadata",
      "durationchange",
      "seeking",
    ];
    reset();
    sync();
    events.forEach((event) => audio.addEventListener(event, sync));
    audio.addEventListener("emptied", reset);
    audio.addEventListener("error", fail);
    return () => {
      events.forEach((event) => audio.removeEventListener(event, sync));
      audio.removeEventListener("emptied", reset);
      audio.removeEventListener("error", fail);
    };
  }, [audioRef, mediaUrl]);

  const toggle = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    cancelActiveTurn();
    setError("");
    if (!audio.paused) audio.pause();
    else {
      try {
        await audio.play();
      } catch {
        setError("Playback could not start. Try playing again.");
      }
    }
  };
  const time = Math.min(position, duration || position);
  return (
    <div className="audio-transport" role="group" aria-label="Audio playback">
      <button
        type="button"
        className="audio-transport-toggle"
        onClick={() => void toggle()}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        aria-label="Playback position"
        aria-valuetext={`${formatTime(time)} of ${duration ? formatTime(duration) : "unknown duration"}`}
        min={0}
        max={duration || 0}
        step={0.1}
        value={duration ? time : 0}
        disabled={!duration}
        onChange={(event) => {
          const audio = audioRef.current;
          if (!audio) return;
          cancelActiveTurn();
          audio.currentTime = Number(event.target.value);
          setPosition(audio.currentTime);
        }}
      />
      <span className="audio-transport-time">
        {formatTime(time)} / {duration ? formatTime(duration) : "—"}
      </span>
      {error && (
        <p className="error audio-transport-error" role="status">
          {error}
        </p>
      )}
    </div>
  );
}

// Speed selector (0.5x-2x). Controlled: the parent owns the rate state and its
// persistence, so both islands stay a single source of truth for the element's
// playbackRate.
export function SpeedControl({
  rate,
  onChange,
}: {
  rate: number;
  onChange: (rate: number) => void;
}) {
  return (
    <label className="text-sm mr-2">
      Speed{" "}
      <select
        className="playback-speed"
        value={rate}
        onChange={(e) => {
          onChange(Number.parseFloat(e.target.value));
        }}
        aria-label="Playback speed"
      >
        {RATE_STEPS.map((step) => (
          <option key={step} value={step}>
            {step}×
          </option>
        ))}
      </select>
    </label>
  );
}

// Codes where the media stream itself is gone (GET /media answers 404/410), so
// even manual scrubbing is impossible — the banner must NOT claim otherwise.
const MEDIA_UNAVAILABLE_CODES = new Set([
  "media_missing",
  "media_reclaimed",
  "media_unservable",
]);

// VISIBLE, honest explanation of why seeking is disabled — never a bare tooltip.
// Renders nothing when seeking is available.
export function CapabilityBanner({
  capability,
}: {
  capability: PlaybackCapability;
}) {
  if (capability.seekEnabled) return null;
  // When the audio stream is present but the timeline can't be trusted, manual
  // scrubbing still works; when the stream itself is unavailable, it does not.
  const mediaGone = capability.reasons.some((r) =>
    MEDIA_UNAVAILABLE_CODES.has(r.code),
  );
  return (
    <div className="notice" role="status" data-testid="capability-banner">
      <p>
        {mediaGone ? (
          <strong>This recording&rsquo;s audio is unavailable.</strong>
        ) : (
          <>
            <strong>Per-segment playback is unavailable for this run.</strong>{" "}
            You can still scrub the audio manually.
          </>
        )}
      </p>
      <ul>
        {capability.reasons.map((reason) => (
          <li key={reason.code}>{reason.message}</li>
        ))}
      </ul>
    </div>
  );
}
