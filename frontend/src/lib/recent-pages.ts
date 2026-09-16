const STORAGE_KEY = "voxint:recent-pages";
const MAX_RECENT = 8;

export interface RecentPage {
  label: string;
  href: string;
}

function isRecentPage(v: unknown): v is RecentPage {
  return (
    v != null &&
    typeof v === "object" &&
    typeof (v as RecentPage).label === "string" &&
    typeof (v as RecentPage).href === "string"
  );
}

export function getRecentPages(): RecentPage[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecentPage).slice(0, MAX_RECENT);
  } catch {
    return [];
  }
}

export function recordPage(label: string, href: string): void {
  try {
    const pages = getRecentPages().filter((p) => p.href !== href);
    pages.unshift({ label, href });
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(pages.slice(0, MAX_RECENT)),
    );
  } catch {
    // sessionStorage unavailable -- silently degrade.
  }
}
