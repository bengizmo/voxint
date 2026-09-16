import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getRecentPages, recordPage } from "./recent-pages";

function createStorage(): Storage {
  const store = new Map<string, string>();
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => {
      store.clear();
    },
    get length() {
      return store.size;
    },
    key: (index: number) => [...store.keys()][index] ?? null,
  };
}

describe("recent-pages", () => {
  beforeEach(() => {
    vi.stubGlobal("sessionStorage", createStorage());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("getRecentPages", () => {
    it("returns empty for fresh session", () => {
      expect(getRecentPages()).toEqual([]);
    });

    it("survives corrupt storage", () => {
      sessionStorage.setItem("voxint:recent-pages", "not json");
      expect(getRecentPages()).toEqual([]);
    });

    it("survives non-array storage", () => {
      sessionStorage.setItem("voxint:recent-pages", '"string"');
      expect(getRecentPages()).toEqual([]);
    });

    it("filters out null entries", () => {
      sessionStorage.setItem(
        "voxint:recent-pages",
        '[null, {"label":"Home","href":"/"}]',
      );
      expect(getRecentPages()).toEqual([{ label: "Home", href: "/" }]);
    });

    it("filters out entries with missing fields", () => {
      sessionStorage.setItem(
        "voxint:recent-pages",
        '[{"label":"Home"}, {"href":"/"}, {"label":"OK","href":"/ok"}]',
      );
      expect(getRecentPages()).toEqual([{ label: "OK", href: "/ok" }]);
    });

    it("filters out entries with wrong types", () => {
      sessionStorage.setItem(
        "voxint:recent-pages",
        '[{"label":42,"href":"/"}, {"label":"OK","href":"/ok"}]',
      );
      expect(getRecentPages()).toEqual([{ label: "OK", href: "/ok" }]);
    });
  });

  describe("recordPage", () => {
    it("adds a page to the front", () => {
      recordPage("Home", "/");
      recordPage("Media", "/media");
      const pages = getRecentPages();
      expect(pages[0]).toEqual({ label: "Media", href: "/media" });
      expect(pages[1]).toEqual({ label: "Home", href: "/" });
    });

    it("deduplicates by href, keeping the latest label", () => {
      recordPage("Home", "/");
      recordPage("Media", "/media");
      recordPage("Home (refreshed)", "/");
      const pages = getRecentPages();
      expect(pages).toHaveLength(2);
      expect(pages[0]).toEqual({ label: "Home (refreshed)", href: "/" });
      expect(pages[1]).toEqual({ label: "Media", href: "/media" });
    });

    it("caps at 8 entries", () => {
      for (let i = 0; i < 12; i++) {
        recordPage(`Page ${i}`, `/page-${i}`);
      }
      const pages = getRecentPages();
      expect(pages).toHaveLength(8);
      expect(pages[0].href).toBe("/page-11");
      expect(pages[7].href).toBe("/page-4");
    });

    it("repairs corrupted storage on next write", () => {
      sessionStorage.setItem(
        "voxint:recent-pages",
        '[null, {"label":"Old","href":"/old"}]',
      );
      recordPage("New", "/new");
      const pages = getRecentPages();
      expect(pages).toEqual([
        { label: "New", href: "/new" },
        { label: "Old", href: "/old" },
      ]);
    });
  });
});
