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

// The JSON shape emitted by the backend `PlaybackCapability.to_props()` (issue
// #55). Shared by every island that gates seeking on it.
export interface CapabilityReason {
  code: string;
  message: string;
}

export interface PlaybackCapability {
  seekEnabled: boolean;
  mediaDuration: number | null;
  reasons: CapabilityReason[];
}

// Discrete speed steps offered by the speed control, within [MIN_RATE, MAX_RATE].
export const RATE_STEPS = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] as const;

interface ActiveTurn {
  audio: HTMLAudioElement;
  onTimeUpdate: () => void;
  rafId: number | null;
}

// Module-level: exactly one guard may be live across the whole page at a time.
let active: ActiveTurn | null = null;

// Tear down the current turn's guard (listener + rAF). Idempotent.
export function cancelActiveTurn(): void {
  if (active === null) return;
  active.audio.removeEventListener("timeupdate", active.onTimeUpdate);
  if (active.rafId !== null) cancelAnimationFrame(active.rafId);
  active = null;
}

/**
 * Seek `audio` to `start` and play until `end`, then pause.
 *
 * No-op on a malformed interval (non-finite, or end <= start) — a fail-closed
 * stance mirroring the backend capability contract: never seek somewhere we
 * cannot bound. The stop boundary is checked against `currentTime` (NOT a
 * pre-computed wall-clock timer, which would overshoot when the speed is raised
 * mid-turn or stop early when it is lowered / the stream buffers): a
 * `requestAnimationFrame` loop gives ~frame-precise stopping in the foreground,
 * and the `timeupdate` listener is the safety net when rAF is throttled (e.g. a
 * backgrounded tab). Both are cancelled together so a fast series of clicks can
 * never leave two guards racing.
 */
export function playTurn(audio: HTMLAudioElement, start: number, end: number): void {
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;

  // Any prior turn's guard must go before we install a new one.
  cancelActiveTurn();

  const stop = (): void => {
    // Only act if THIS turn is still the active one (a newer turn may have
    // superseded it between the check firing and this callback running).
    if (active === null || active.audio !== audio) return;
    cancelActiveTurn();
    audio.pause();
  };

  const onTimeUpdate = (): void => {
    if (audio.currentTime >= end) stop();
  };

  const tick = (): void => {
    if (active === null || active.audio !== audio) return;
    if (audio.currentTime >= end) {
      stop();
      return;
    }
    active.rafId = requestAnimationFrame(tick);
  };

  active = { audio, onTimeUpdate, rafId: null };
  audio.addEventListener("timeupdate", onTimeUpdate);
  active.rafId = requestAnimationFrame(tick);

  audio.currentTime = start;
  const played = audio.play();
  // play() rejects if the browser blocks autoplay or the element is torn down
  // mid-call; swallow it so an unhandled rejection never surfaces to the user.
  // The guard is cancelled by the next playTurn / cancelActiveTurn regardless.
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
