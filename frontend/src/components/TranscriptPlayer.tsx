import { useEffect, useRef, useState } from "react";

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
}

function formatTime(seconds: number): string {
  return seconds.toFixed(2);
}

// Read-only, audio-synced transcript. Native <audio> plus the segment list,
// highlighting the currently-playing segment via the element's `timeupdate`
// event. NO click-to-seek, NO relabeling, NO keyboard shortcuts — those are
// #49+.
export function TranscriptPlayer({ runId, mediaUrl, segments }: TranscriptPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [activeIndex, setActiveIndex] = useState<number>(-1);

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
      // On unmount, stop the stream so switching runs can't leave it open.
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    };
  }, [segments]);

  return (
    <div>
      <audio ref={audioRef} controls src={mediaUrl} className="w-full my-2" data-run-id={runId}>
        Your browser does not support the audio element.
      </audio>
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
          >
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
