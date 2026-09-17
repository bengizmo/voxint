/**
 * Shared helpers for combobox widgets (speaker combobox, command palette).
 *
 * Pure functions, no side effects, tested with vitest.
 */

/**
 * Rank items by how well `getLabel(item)` matches `query`.
 *
 * Three tiers (stable within each):
 * 1. Label starts with the query
 * 2. A word in the label starts with the query
 * 3. Label contains the query as a substring
 *
 * Returns all matches when `cap` is omitted; pass a number to truncate.
 */
export function rankItems<T>(
  items: readonly T[],
  query: string,
  getLabel: (item: T) => string,
  cap?: number,
): T[] {
  const q = query.toLowerCase().trim();
  if (!q) return cap != null ? items.slice(0, cap) : [...items];

  const prefix: T[] = [];
  const wordPrefix: T[] = [];
  const substring: T[] = [];

  for (const item of items) {
    const lower = getLabel(item).toLowerCase();
    if (lower.startsWith(q)) {
      prefix.push(item);
    } else if (lower.split(/\s+/).some((w) => w.startsWith(q))) {
      wordPrefix.push(item);
    } else if (lower.includes(q)) {
      substring.push(item);
    }
  }

  const result = [...prefix, ...wordPrefix, ...substring];
  return cap != null ? result.slice(0, cap) : result;
}

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
