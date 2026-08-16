// Shared per-turn playback primitives (issue #49), used by BOTH the in-React
// transcript player and the workbench island's document-delegated buttons.
//
// The contract: one active "turn" at a time. Starting a new turn (or unmounting)
// cancels the previous turn's end-guard so a fast series of clicks can never
// leave two guards racing to pause the element — which would otherwise cut a
// later turn short or overshoot into the next voice.

export const MIN_RATE = 0.5;
export const MAX_RATE = 2.0;
const RATE_KEY = "voxint.playbackRate";

interface ActiveTurn {
  audio: HTMLAudioElement;
  onTimeUpdate: () => void;
  timer: ReturnType<typeof setTimeout>;
}

// Module-level: exactly one guard may be live across the whole page at a time.
let active: ActiveTurn | null = null;

// Tear down the current turn's guard (listener + timer). Idempotent.
export function cancelActiveTurn(): void {
  if (active === null) return;
  active.audio.removeEventListener("timeupdate", active.onTimeUpdate);
  clearTimeout(active.timer);
  active = null;
}

/**
 * Seek `audio` to `start` and play until `end`, then pause.
 *
 * No-op on a malformed interval (non-finite, or end <= start) — a fail-closed
 * stance mirroring the backend capability contract: never seek somewhere we
 * cannot bound. Stopping at `end` combines a one-shot `timeupdate` check with a
 * wall-clock `setTimeout` fallback scaled by the current playbackRate, because a
 * coarse timeupdate cadence alone can overshoot past `end` into the next
 * speaker's audio before it fires.
 */
export function playTurn(audio: HTMLAudioElement, start: number, end: number): void {
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;

  // Any prior turn's guard must go before we install a new one.
  cancelActiveTurn();

  const stop = (): void => {
    // Only act if THIS turn is still the active one (a newer turn may have
    // superseded it between the event firing and this callback running).
    if (active === null || active.audio !== audio) return;
    cancelActiveTurn();
    audio.pause();
  };

  const onTimeUpdate = (): void => {
    if (audio.currentTime >= end) stop();
  };

  const rate = audio.playbackRate > 0 ? audio.playbackRate : 1;
  // A little slack (50ms) past the computed span so the fallback only fires when
  // timeupdate genuinely failed to, not in a photo-finish with it.
  const fallbackMs = ((end - start) / rate) * 1000 + 50;
  const timer = setTimeout(stop, fallbackMs);

  active = { audio, onTimeUpdate, timer };
  audio.addEventListener("timeupdate", onTimeUpdate);

  audio.currentTime = start;
  const played = audio.play();
  // play() rejects if the browser blocks autoplay or the element is torn down
  // mid-call; swallow it so an unhandled rejection never surfaces to the user.
  if (played && typeof played.catch === "function") {
    played.catch(() => {
      /* playback blocked or interrupted; the guard still cleans up */
    });
  }
}

export function clampRate(rate: number): number {
  if (!Number.isFinite(rate)) return 1;
  return Math.min(MAX_RATE, Math.max(MIN_RATE, rate));
}

// Persisted playback rate. Storage may be unavailable (private mode, disabled
// cookies); every access degrades to the 1.0 default rather than throwing.
export function getStoredRate(): number {
  try {
    const raw = localStorage.getItem(RATE_KEY);
    if (raw === null) return 1;
    const parsed = Number.parseFloat(raw);
    return Number.isFinite(parsed) ? clampRate(parsed) : 1;
  } catch {
    return 1;
  }
}

export function setStoredRate(rate: number): void {
  try {
    localStorage.setItem(RATE_KEY, String(clampRate(rate)));
  } catch {
    /* storage unavailable; the in-memory rate still applies for this session */
  }
}
