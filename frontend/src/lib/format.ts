export function formatTime(
  seconds: number,
  opts?: { long?: boolean; decimals?: number },
): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";

  const d = opts?.decimals;
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  const secStr =
    d != null && d > 0
      ? (seconds % 60).toFixed(d).padStart(d + 3, "0")
      : String(secs).padStart(2, "0");

  if (opts?.long || hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${secStr}`;
  }
  return `${minutes}:${secStr}`;
}
