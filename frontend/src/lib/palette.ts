/**
 * Pure helpers for the command palette island (#162).
 *
 * All functions are side-effect-free and tested with vitest.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PaletteCommand {
  label: string;
  href: string;
  kind: "destination" | "action";
  hint: string | null;
}

export interface EntityItem {
  kind: "media" | "speaker" | "project";
  id: string;
  label: string;
  sublabel: string | null;
  href: string;
  archived: boolean;
}

export interface PassageItem {
  run_id: string;
  title: string;
  speaker_label: string | null;
  start_seconds: number;
  snippet: string;
  href: string;
}

export type PassageState =
  | "ok"
  | "empty_query"
  | "off"
  | "unavailable"
  | "indexing"
  | "short_query"
  | "loading"
  | "idle";

export interface Row {
  id: string;
  type: "command" | "entity" | "passage";
  label: string;
  sublabel: string | null;
  href: string;
}

export interface RowGroup {
  label: string;
  rows: Row[];
}

// ---------------------------------------------------------------------------
// Chord detection
// ---------------------------------------------------------------------------

/**
 * Returns true when the keyboard event is the palette open/toggle chord.
 * Checks Ctrl+K (Linux/Windows) or Cmd+K (macOS). Returns false when the
 * event is a key repeat, during IME composition, or when Alt is held.
 */
export function isPaletteChord(event: KeyboardEvent): boolean {
  if (event.repeat || event.isComposing) return false;
  if (event.altKey) return false;
  if (!(event.ctrlKey || event.metaKey)) return false;
  // Shift+Ctrl+K in Firefox is "Web Developer Tools"; ignore shift.
  if (event.shiftKey) return false;
  return event.key.toLowerCase() === "k";
}

// ---------------------------------------------------------------------------
// Command filtering
// ---------------------------------------------------------------------------

/**
 * Filter and rank commands against a query string. Ranking:
 * 1. Label starts with the query (case-insensitive)
 * 2. A word in the label starts with the query
 * 3. Label contains the query as a substring
 *
 * Stable within each tier. Capped at `cap` results.
 */
export function filterCommands(
  commands: readonly PaletteCommand[],
  query: string,
  cap = 6,
): PaletteCommand[] {
  const q = query.toLowerCase().trim();
  if (!q) return commands.slice(0, cap);

  const prefix: PaletteCommand[] = [];
  const wordPrefix: PaletteCommand[] = [];
  const substring: PaletteCommand[] = [];

  for (const cmd of commands) {
    const lower = cmd.label.toLowerCase();
    if (lower.startsWith(q)) {
      prefix.push(cmd);
    } else if (
      lower
        .split(/\s+/)
        .some((w) => w.startsWith(q))
    ) {
      wordPrefix.push(cmd);
    } else if (lower.includes(q)) {
      substring.push(cmd);
    }
  }

  return [...prefix, ...wordPrefix, ...substring].slice(0, cap);
}

// ---------------------------------------------------------------------------
// Row building
// ---------------------------------------------------------------------------

/**
 * Build a flat indexed row list from the three result groups. Each row gets
 * a stable `id` for `aria-activedescendant`.
 */
export function buildRows(
  commands: readonly PaletteCommand[],
  entities: readonly EntityItem[],
  passages: readonly PassageItem[],
): RowGroup[] {
  const groups: RowGroup[] = [];
  let idx = 0;

  if (commands.length > 0) {
    groups.push({
      label: "Commands",
      rows: commands.map((c) => ({
        id: `palette-opt-${idx++}`,
        type: "command" as const,
        label: c.label,
        sublabel: c.hint,
        href: c.href,
      })),
    });
  }

  if (entities.length > 0) {
    groups.push({
      label: "Results",
      rows: entities.map((e) => ({
        id: `palette-opt-${idx++}`,
        type: "entity" as const,
        label: e.label,
        sublabel: e.sublabel,
        href: e.href,
      })),
    });
  }

  if (passages.length > 0) {
    groups.push({
      label: "Transcripts",
      rows: passages.map((p) => ({
        id: `palette-opt-${idx++}`,
        type: "passage" as const,
        label: p.snippet || p.title,
        sublabel: p.speaker_label
          ? `${p.title} — ${p.speaker_label}`
          : p.title,
        href: p.href,
      })),
    });
  }

  return groups;
}

/**
 * Count all rows across all groups.
 */
export function totalRows(groups: readonly RowGroup[]): number {
  return groups.reduce((sum, g) => sum + g.rows.length, 0);
}

/**
 * Get the flat row at a given index across groups.
 */
export function rowAt(
  groups: readonly RowGroup[],
  index: number,
): Row | undefined {
  let offset = 0;
  for (const group of groups) {
    if (index < offset + group.rows.length) {
      return group.rows[index - offset];
    }
    offset += group.rows.length;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Keyboard navigation
// ---------------------------------------------------------------------------

/**
 * Move the active index for ArrowUp/Down/Home/End. Clamped, no wrap.
 */
export function moveActive(
  index: number,
  key: string,
  count: number,
): number {
  if (count === 0) return -1;
  switch (key) {
    case "ArrowDown":
      return Math.min(index + 1, count - 1);
    case "ArrowUp":
      return Math.max(index - 1, 0);
    case "Home":
      return 0;
    case "End":
      return count - 1;
    default:
      return index;
  }
}

// ---------------------------------------------------------------------------
// Query normalization
// ---------------------------------------------------------------------------

/**
 * Normalize and trim a query string. Collapses internal whitespace.
 */
export function normalizeQuery(raw: string): string {
  return raw.trim().replace(/\s+/g, " ");
}

// ---------------------------------------------------------------------------
// Passage state copy
// ---------------------------------------------------------------------------

/** Human-readable copy for each passage search state. */
export const PASSAGE_STATE_COPY: Record<PassageState, string> = {
  ok: "",
  empty_query: "",
  short_query: "",
  idle: "",
  loading: "Searching transcripts…",
  off: "Transcript search is not enabled.",
  unavailable: "Transcript search is not available right now.",
  indexing: "Transcript index is still building.",
};
