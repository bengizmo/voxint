// Navigable outline (issue #87): the shapes the server ships in
// island_props.outline and small pure helpers the panel and its tests share.
//
// The panel resolves each target's startSeconds to a current rendered line at
// click time (see resolveJumpIndex in ./jump); the server never bakes a line
// index, because segment ordinals diverge from rendered lines after a split.

export interface OutlineOccurrence {
  segmentIndex: number;
  startSeconds: number;
  quote: string;
}

export interface OutlineMention {
  surface: string;
  kind: string | null;
  occurrences: OutlineOccurrence[];
}

export interface OutlineContext {
  summary: string | null;
  topics: string[];
}

export interface OutlineDiagnostics {
  droppedUnlocatable: number;
  droppedOutOfRun: number;
  droppedUnresolved: number;
}

export interface OutlineProps {
  // Whether an entity_mentions asset exists at all. false with gated=true means
  // the feature is off; false with gated=false means none was generated yet.
  available: boolean;
  gated: boolean;
  // The source text changed since generation: one panel banner, never a seek block.
  assetStale: boolean;
  mentions: OutlineMention[];
  context: OutlineContext;
  diagnostics: OutlineDiagnostics;
}

// mm:ss (or h:mm:ss past an hour). Coarse by design: the target is a segment
// start, not a word, so no sub-second precision is implied.
export function formatClock(totalSeconds: number): string {
  const clamped = Number.isFinite(totalSeconds) && totalSeconds > 0 ? totalSeconds : 0;
  const whole = Math.floor(clamped);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const seconds = whole % 60;
  const mm = hours > 0 ? String(minutes).padStart(2, "0") : String(minutes);
  const ss = String(seconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function totalDropped(diagnostics: OutlineDiagnostics): number {
  return (
    diagnostics.droppedUnlocatable +
    diagnostics.droppedOutOfRun +
    diagnostics.droppedUnresolved
  );
}
