// Waveform-strip data plumbing (issue #57): the peaks payload contract with
// GET /media/{run_id}/peaks, the diarization-turn region shape, and the
// click→segment mapping. Pure functions — the canvas drawing lives in
// WaveformStrip.tsx.

import { apiFetch } from "./api-client";
import type { Segment } from "../components/TranscriptPlayer";

// One diarization turn, as serialized into island props by the server. Turns —
// not transcript segments — are what the strip PAINTS: they are the honest
// who-spoke-when record (dense, include untranscribed speech, carry overlap).
export interface Turn {
  start: number;
  end: number;
  // Same curated palette index the list badges use (null = label outside the
  // run's canonical universe → neutral fill).
  paletteIndex: number | null;
  // True where the diarizer marked this turn as overlapping another speaker;
  // the strip renders these with a distinct marker (honest "two voices here").
  overlap: boolean;
}

export interface PeaksPayload {
  version: number;
  // Authoritative time→x axis for the strip (measured from the WAV itself);
  // deliberately NOT capability.mediaDuration, which can be null/invalid while
  // a cached envelope is still trustworthy.
  duration: number;
  sampleRate: number;
  frameCount: number;
  samplesPerBucket: number;
  // max(|sample|)/32768 per fixed bucket, 0..1.
  peaks: number[];
}

// Validate an untrusted response body. Anything off-contract returns null and
// the strip simply does not render — honest degradation, never a broken axis.
export function parsePeaksPayload(data: unknown): PeaksPayload | null {
  if (typeof data !== "object" || data === null) return null;
  const d = data as Record<string, unknown>;
  if (d.version !== 1) return null;
  if (
    typeof d.duration !== "number" ||
    !Number.isFinite(d.duration) ||
    d.duration <= 0
  )
    return null;
  if (typeof d.sampleRate !== "number" || d.sampleRate <= 0) return null;
  if (typeof d.frameCount !== "number" || d.frameCount <= 0) return null;
  if (typeof d.samplesPerBucket !== "number" || d.samplesPerBucket <= 0)
    return null;
  if (!Array.isArray(d.peaks) || d.peaks.length === 0) return null;
  if (!d.peaks.every((v) => typeof v === "number" && Number.isFinite(v)))
    return null;
  return d as unknown as PeaksPayload;
}

// Map a strip-click time to the transcript segment containing it — the SAME
// containment rule as the follow-along timeupdate (t in [start, end)), so a
// click lands exactly where the highlight would be at that moment. A click in
// a gap (diarized but untranscribed speech, silence) returns -1: nothing to
// select in the list, deliberate no-op. Linear scan: 500–2000 segments is
// nothing per click, and it preserves the first-match-wins overlap semantics.
export function segmentAtTime(segments: Segment[], t: number): number {
  return segments.findIndex((seg) => t >= seg.start && t < seg.end);
}

// Fetch + validate the envelope. ANY failure (404/410, network, off-contract
// body) resolves null: the strip is pure enhancement and its absence needs no
// error surface. No retry — the server result will not change under a loop.
export async function fetchPeaks(
  url: string,
  signal: AbortSignal,
): Promise<PeaksPayload | null> {
  try {
    const res = await apiFetch(url, {
      headers: { accept: "application/json" },
      signal,
    });
    return parsePeaksPayload((await res.json()) as unknown);
  } catch {
    return null;
  }
}
