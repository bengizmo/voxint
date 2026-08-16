import { RATE_STEPS, type PlaybackCapability } from "../lib/playback";

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

// VISIBLE, honest explanation of why seeking is disabled — never a bare tooltip.
// Renders nothing when seeking is available.
export function CapabilityBanner({ capability }: { capability: PlaybackCapability }) {
  if (capability.seekEnabled) return null;
  return (
    <div className="notice" role="status" data-testid="capability-banner">
      <p>
        <strong>Per-segment playback is unavailable for this run.</strong> You can still
        scrub the audio manually.
      </p>
      <ul>
        {capability.reasons.map((reason) => (
          <li key={reason.code}>{reason.message}</li>
        ))}
      </ul>
    </div>
  );
}
