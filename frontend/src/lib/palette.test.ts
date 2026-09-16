import { describe, expect, it } from "vitest";

import {
  type PaletteCommand,
  type PassageState,
  PASSAGE_STATE_COPY,
  buildRows,
  filterCommands,
  isPaletteChord,
  moveActive,
  normalizeQuery,
  rowAt,
  totalRows,
} from "./palette";

// ---------------------------------------------------------------------------
// isPaletteChord
// ---------------------------------------------------------------------------

describe("isPaletteChord", () => {
  const make = (overrides: Partial<KeyboardEvent> = {}): KeyboardEvent =>
    ({
      key: "k",
      ctrlKey: true,
      metaKey: false,
      altKey: false,
      shiftKey: false,
      repeat: false,
      isComposing: false,
      ...overrides,
    }) as unknown as KeyboardEvent;

  it("accepts Ctrl+K", () => {
    expect(isPaletteChord(make())).toBe(true);
  });

  it("accepts Cmd+K (macOS)", () => {
    expect(isPaletteChord(make({ ctrlKey: false, metaKey: true }))).toBe(true);
  });

  it("rejects when repeat", () => {
    expect(isPaletteChord(make({ repeat: true }))).toBe(false);
  });

  it("rejects during IME composition", () => {
    expect(isPaletteChord(make({ isComposing: true }))).toBe(false);
  });

  it("rejects Alt+K", () => {
    expect(isPaletteChord(make({ altKey: true }))).toBe(false);
  });

  it("rejects Shift+Ctrl+K", () => {
    expect(isPaletteChord(make({ shiftKey: true }))).toBe(false);
  });

  it("rejects plain K (no modifier)", () => {
    expect(isPaletteChord(make({ ctrlKey: false }))).toBe(false);
  });

  it("rejects Ctrl+J", () => {
    expect(isPaletteChord(make({ key: "j" }))).toBe(false);
  });

  it("is case-insensitive on the key", () => {
    expect(isPaletteChord(make({ key: "K" }))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// filterCommands
// ---------------------------------------------------------------------------

describe("filterCommands", () => {
  const cmds: PaletteCommand[] = [
    { label: "Home", href: "/", kind: "destination", hint: null },
    { label: "Media", href: "/media", kind: "destination", hint: null },
    { label: "Add media", href: "/media", kind: "action", hint: null },
    { label: "Settings", href: "/settings", kind: "destination", hint: null },
    { label: "Speakers", href: "/speakers", kind: "destination", hint: null },
  ];

  it("returns all commands (capped) with empty query", () => {
    const result = filterCommands(cmds, "", 3);
    expect(result).toHaveLength(3);
    expect(result[0].label).toBe("Home");
  });

  it("ranks prefix matches first", () => {
    const result = filterCommands(cmds, "me");
    expect(result[0].label).toBe("Media");
  });

  it("ranks word-prefix matches after prefix", () => {
    const result = filterCommands(cmds, "media");
    // "Media" is prefix, "Add media" is word-prefix
    expect(result[0].label).toBe("Media");
    expect(result[1].label).toBe("Add media");
  });

  it("ranks substring matches last", () => {
    const result = filterCommands(cmds, "etting");
    expect(result[0].label).toBe("Settings");
  });

  it("is case-insensitive", () => {
    const result = filterCommands(cmds, "HOME");
    expect(result[0].label).toBe("Home");
  });

  it("returns empty for no matches", () => {
    expect(filterCommands(cmds, "zzz")).toHaveLength(0);
  });

  it("trims the query", () => {
    const result = filterCommands(cmds, "  home  ");
    expect(result[0].label).toBe("Home");
  });
});

// ---------------------------------------------------------------------------
// buildRows / totalRows / rowAt
// ---------------------------------------------------------------------------

describe("buildRows", () => {
  const cmds: PaletteCommand[] = [
    { label: "Home", href: "/", kind: "destination", hint: null },
    { label: "Media", href: "/media", kind: "destination", hint: null },
  ];

  it("creates a Commands group from commands", () => {
    const groups = buildRows(cmds, [], []);
    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("Commands");
    expect(groups[0].rows).toHaveLength(2);
  });

  it("assigns sequential ids", () => {
    const groups = buildRows(cmds, [], []);
    expect(groups[0].rows[0].id).toBe("palette-opt-0");
    expect(groups[0].rows[1].id).toBe("palette-opt-1");
  });

  it("returns empty for empty inputs", () => {
    expect(buildRows([], [], [])).toHaveLength(0);
  });

  it("totalRows counts across groups", () => {
    const groups = buildRows(cmds, [], []);
    expect(totalRows(groups)).toBe(2);
  });

  it("rowAt returns the correct row by flat index", () => {
    const groups = buildRows(cmds, [], []);
    expect(rowAt(groups, 0)?.label).toBe("Home");
    expect(rowAt(groups, 1)?.label).toBe("Media");
    expect(rowAt(groups, 2)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// moveActive
// ---------------------------------------------------------------------------

describe("moveActive", () => {
  it("ArrowDown increments", () => {
    expect(moveActive(0, "ArrowDown", 5)).toBe(1);
  });

  it("ArrowDown clamps at last", () => {
    expect(moveActive(4, "ArrowDown", 5)).toBe(4);
  });

  it("ArrowUp decrements", () => {
    expect(moveActive(2, "ArrowUp", 5)).toBe(1);
  });

  it("ArrowUp clamps at zero", () => {
    expect(moveActive(0, "ArrowUp", 5)).toBe(0);
  });

  it("Home goes to zero", () => {
    expect(moveActive(3, "Home", 5)).toBe(0);
  });

  it("End goes to last", () => {
    expect(moveActive(0, "End", 5)).toBe(4);
  });

  it("returns -1 for empty list", () => {
    expect(moveActive(0, "ArrowDown", 0)).toBe(-1);
  });

  it("ignores unknown keys", () => {
    expect(moveActive(2, "x", 5)).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// normalizeQuery
// ---------------------------------------------------------------------------

describe("normalizeQuery", () => {
  it("trims whitespace", () => {
    expect(normalizeQuery("  hello  ")).toBe("hello");
  });

  it("collapses internal whitespace", () => {
    expect(normalizeQuery("hello   world")).toBe("hello world");
  });
});

// ---------------------------------------------------------------------------
// PASSAGE_STATE_COPY
// ---------------------------------------------------------------------------

describe("PASSAGE_STATE_COPY", () => {
  it("has an entry for every PassageState value", () => {
    const states: PassageState[] = [
      "ok",
      "empty_query",
      "short_query",
      "idle",
      "loading",
      "off",
      "unavailable",
      "indexing",
    ];
    for (const s of states) {
      expect(PASSAGE_STATE_COPY).toHaveProperty(s);
    }
  });
});
