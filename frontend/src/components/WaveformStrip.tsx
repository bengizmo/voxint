import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import type { PeaksPayload, Turn } from "../lib/peaks";
import { segmentAtTime } from "../lib/peaks";
import type { Segment } from "./TranscriptPlayer";

// Who-spoke-when waveform strip (issue #57). One DPR-aware canvas: mirrored
// amplitude bars from the precomputed peaks, tinted per DIARIZATION TURN with
// the same `--spk-accent` palette the list badges use (turns are the honest
// speaker record; transcript segments only carry a dominant label). A click
// maps time → transcript segment and reports it upward — the strip itself
// NEVER touches the audio element, so the fail-closed seek gate (issue #55)
// stays structural: the only seek path is the caller's gated playTurn.
//
// aria-hidden by design: every action here (select / play a segment) exists in
// the accessible list with real buttons; exposing 2000 canvas regions to AT
// would be noise, not access. The data-* attributes are for the E2E lane.

interface WaveformStripProps {
  peaks: PeaksPayload;
  turns: Turn[];
  segments: Segment[];
  // Playback follow-along emphasis (index into `segments`, -1 when none).
  activeIndex: number;
  // Review-cursor marker (underline). Absent on the read-only surface.
  cursorIndex?: number;
  seekEnabled: boolean;
  // Quantized playhead position (seconds). Hidden when seeking is untrusted —
  // a playback affordance would over-promise on an unreliable timeline.
  currentTime: number;
  onRegionActivate: (index: number) => void;
}

const STRIP_HEIGHT = 72;
const PALETTE_SIZE = 8; // mirrors speaker_colors.PALETTE_SIZE
// Last-resort default only: the LIVE neutral is read from the --wave-neutral
// token at draw time (see resolveNeutral) so the canvas tracks the stylesheet,
// like the speaker accents below. This literal is reached only if the token is
// absent (never in the app shell, where base.html defines it in :root).
const NEUTRAL_BAR = "#88888c";

// The canvas has no stylesheet of its own, so it reads the neutral bar color
// from the shell's --wave-neutral token (issue #90) rather than hardcoding it —
// the root owns this global token, so a direct lookup is enough.
function resolveNeutral(): string {
  return (
    getComputedStyle(document.documentElement)
      .getPropertyValue("--wave-neutral")
      .trim() || NEUTRAL_BAR
  );
}

// Resolve the run palette's CSS accents from probe spans so the canvas uses
// EXACTLY the colors the stylesheet (light or dark) currently resolves to.
function resolveAccents(probes: HTMLElement | null, neutral: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < PALETTE_SIZE; i += 1) {
    const el = probes?.querySelector<HTMLElement>(`.spk-${i}`);
    const v = el
      ? getComputedStyle(el).getPropertyValue("--spk-accent").trim()
      : "";
    out.push(v || neutral);
  }
  return out;
}

export function WaveformStrip({
  peaks,
  turns,
  segments,
  activeIndex,
  cursorIndex,
  seekEnabled,
  currentTime,
  onRegionActivate,
}: WaveformStripProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const probesRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(0);
  // Bumped when the color scheme flips so the draw effect re-resolves accents.
  const [themeEpoch, setThemeEpoch] = useState<number>(0);
  // Left position (%) of the "no transcript here" note after a click that lands
  // on no segment; null hides it. Cleared by the next valid click, replaced by
  // the next gap click (see onClick). No timer / toast framework — the note is
  // strip-local and purely presentational, so it never touches the seek gate.
  const [gapHint, setGapHint] = useState<number | null>(null);

  // Track the rendered width (ResizeObserver, not window resize: the strip
  // lives in a variable-width column).
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      setWidth(Math.round(w));
    });
    observer.observe(canvas);
    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      setThemeEpoch((n) => n + 1);
    };
    mq.addEventListener("change", onChange);
    return () => {
      mq.removeEventListener("change", onChange);
    };
  }, []);

  // Full repaint. Runs only on mount / resize / theme flip / emphasis change —
  // never per playback frame (the playhead is a separately-positioned div).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(STRIP_HEIGHT * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, STRIP_HEIGHT);

    const neutral = resolveNeutral();
    const accents = resolveAccents(probesRef.current, neutral);
    const accentOf = (paletteIndex: number | null): string =>
      paletteIndex != null && paletteIndex >= 0 && paletteIndex < accents.length
        ? accents[paletteIndex]
        : neutral;
    const { duration } = peaks;
    const xOf = (t: number): number =>
      Math.max(0, Math.min(width, (t / duration) * width));
    const mid = STRIP_HEIGHT / 2;
    const buckets = peaks.peaks.length;

    // 1. Per-bucket ownership — ONE source of truth for both the backdrop and
    //    the bars, so the two passes can never disagree around overlaps. Each
    //    bucket is colored by the LAST turn (in the server's (start, turn_index)
    //    order) whose interval intersects it (floor/ceil span, not center — a
    //    bucket that contains any of a turn's speech reads as that speaker);
    //    later turns overwrite earlier, so "later wins" holds deterministically.
    const barW = width / buckets;
    const bucketColor: string[] = new Array<string>(buckets).fill(neutral);
    const covered: boolean[] = new Array<boolean>(buckets).fill(false);
    for (const turn of turns) {
      const b0 = Math.max(0, Math.floor((turn.start / duration) * buckets));
      const b1 = Math.min(buckets, Math.ceil((turn.end / duration) * buckets));
      const color = accentOf(turn.paletteIndex);
      for (let b = b0; b < b1; b += 1) {
        bucketColor[b] = color;
        covered[b] = true;
      }
    }

    // 2. Region backdrops: a faint full-height tint drawn from that same
    //    ownership as run-length rects (consecutive same-color buckets), so the
    //    backdrop matches the bars exactly and abutting rects never double-blend
    //    their alpha. Regions read even during silence.
    ctx.globalAlpha = 0.16;
    for (let b = 0; b < buckets; ) {
      if (!covered[b]) {
        b += 1;
        continue;
      }
      const color = bucketColor[b];
      let e = b + 1;
      while (e < buckets && covered[e] && bucketColor[e] === color) e += 1;
      ctx.fillStyle = color;
      ctx.fillRect(b * barW, 0, (e - b) * barW, STRIP_HEIGHT);
      b = e;
    }

    // 3. Bars: mirrored amplitude, colored by the same ownership. The width is
    //    capped at barW so a sub-pixel barW (a strip under ~1000px with 2000
    //    buckets) can't overdraw its neighbor and band the translucent fills.
    const drawBarW = Math.min(Math.max(barW * 0.8, 0.5), barW);
    for (let b = 0; b < buckets; b += 1) {
      const amp = Math.max(peaks.peaks[b] * (mid - 2), 0.75);
      ctx.globalAlpha = covered[b] ? 0.9 : 0.45;
      ctx.fillStyle = bucketColor[b];
      ctx.fillRect(b * barW, mid - amp, drawBarW, amp * 2);
    }

    // 4. Overlap marker: diagonal hatching where the diarizer flagged two
    //    voices — an honest "not one speaker here" cue. Drawn per-turn (not from
    //    bucket ownership) so a hidden earlier speaker still shows it overlapped.
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = neutral;
    ctx.lineWidth = 1;
    for (const turn of turns) {
      if (!turn.overlap) continue;
      const x0 = xOf(turn.start);
      const x1 = xOf(turn.end);
      if (x1 <= x0) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, 0, x1 - x0, STRIP_HEIGHT);
      ctx.clip();
      ctx.beginPath();
      for (let x = x0 - STRIP_HEIGHT; x < x1; x += 6) {
        ctx.moveTo(x, STRIP_HEIGHT);
        ctx.lineTo(x + STRIP_HEIGHT, 0);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 5. Emphasis from the LIST's state (segments, not turns): the playing
    //    segment gets a brighter full-height tint; the review cursor gets a
    //    3px underline in its accent.
    const active = activeIndex >= 0 ? segments[activeIndex] : undefined;
    if (active) {
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = accentOf(active.paletteIndex);
      ctx.fillRect(
        xOf(active.start),
        0,
        Math.max(xOf(active.end) - xOf(active.start), 1),
        STRIP_HEIGHT,
      );
    }
    const cursor =
      cursorIndex != null && cursorIndex >= 0
        ? segments[cursorIndex]
        : undefined;
    if (cursor) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = accentOf(cursor.paletteIndex);
      ctx.fillRect(
        xOf(cursor.start),
        STRIP_HEIGHT - 3,
        Math.max(xOf(cursor.end) - xOf(cursor.start), 2),
        3,
      );
    }
    ctx.globalAlpha = 1;
  }, [peaks, turns, segments, activeIndex, cursorIndex, width, themeEpoch]);

  const onClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || width <= 0) return;
      const rect = canvas.getBoundingClientRect();
      const relX = (event.clientX - rect.left) / rect.width;
      const index = segmentAtTime(segments, relX * peaks.duration);
      if (index >= 0) {
        setGapHint(null);
        onRegionActivate(index);
        return;
      }
      // No transcript segment covers this point (untranscribed speech OR
      // silence — segmentAtTime cannot tell them apart, so the note must not
      // claim either). A silent no-op on a click-to-play affordance reads as a
      // broken control to a non-technical operator; a brief strip-local note
      // makes the limit honest. Presentational only: no upward callback, no
      // snapping to a nearby segment, no audio — the seek gate is untouched.
      setGapHint(Math.min(Math.max(relX * 100, 0), 100));
    },
    [segments, peaks.duration, width, onRegionActivate],
  );

  // currentTime is -1 until the first timeupdate; >= 0 shows the playhead even
  // while segment 0 (start = 0s) plays. Still hidden when seeking is untrusted.
  const playheadPct =
    seekEnabled && Number.isFinite(currentTime) && currentTime >= 0
      ? Math.min((currentTime / peaks.duration) * 100, 100)
      : null;

  return (
    <div
      aria-hidden="true"
      data-testid="waveform-strip"
      data-active-index={activeIndex}
      data-cursor-index={cursorIndex ?? -1}
      className="relative w-full my-2 rounded border border-line/40 overflow-hidden"
      style={{ height: STRIP_HEIGHT }}
      title={
        seekEnabled
          ? "Who spoke when — click to play that part"
          : "Who spoke when — seeking is unavailable, so clicking only shows the segment in the list"
      }
    >
      {/* Palette probes: resolve the stylesheet's current --spk-accent values
          (light/dark) without duplicating them in JS. */}
      <div ref={probesRef} className="hidden">
        {Array.from({ length: PALETTE_SIZE }, (_, i) => (
          <span key={i} className={`spk-${i}`} />
        ))}
      </div>
      <canvas
        ref={canvasRef}
        onClick={onClick}
        className="block w-full h-full"
        style={{ cursor: "pointer" }}
      />
      {playheadPct != null && (
        <div
          data-testid="waveform-playhead"
          className="absolute top-0 bottom-0 w-[1.5px] bg-current pointer-events-none"
          style={{ left: `${playheadPct}%` }}
        />
      )}
      {gapHint != null && (
        <>
          {/* Marker at the click, so the centered note (below) still reads
              which point had no transcript without risking edge clipping. */}
          <div
            className="absolute top-0 bottom-0 w-px bg-line pointer-events-none"
            style={{ left: `${gapHint}%` }}
          />
          <div
            data-testid="waveform-gap-hint"
            className="absolute left-1/2 top-1 -translate-x-1/2 px-1.5 py-0.5 rounded border border-line text-[11px] leading-none whitespace-nowrap pointer-events-none"
            style={{ background: "var(--surface)", color: "var(--ink-2)" }}
          >
            No transcript text at this point
          </div>
        </>
      )}
    </div>
  );
}
