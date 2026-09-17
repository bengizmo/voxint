import { describe, expect, it } from "vitest";

import { moveActive, rankItems } from "./combobox";

describe("rankItems", () => {
  const items = ["Alice", "Bob", "Charlie", "Alicia", "Bobby"];

  it("returns all items when query is empty", () => {
    expect(rankItems(items, "", (s) => s)).toEqual(items);
  });

  it("returns all items when query is whitespace", () => {
    expect(rankItems(items, "  ", (s) => s)).toEqual(items);
  });

  it("ranks prefix matches first", () => {
    const result = rankItems(items, "bo", (s) => s);
    expect(result).toEqual(["Bob", "Bobby"]);
  });

  it("ranks word-prefix matches after full-prefix", () => {
    const items = ["Bob Smith", "Alice Bob", "Bobcat Jones"];
    const result = rankItems(items, "bob", (s) => s);
    expect(result[0]).toBe("Bob Smith");
    expect(result[1]).toBe("Bobcat Jones");
    expect(result[2]).toBe("Alice Bob");
  });

  it("ranks substring matches last", () => {
    const items = ["Jacoby", "Bob"];
    const result = rankItems(items, "ob", (s) => s);
    expect(result).toEqual(["Jacoby", "Bob"]);
  });

  it("is case insensitive", () => {
    expect(rankItems(items, "ALICE", (s) => s)).toEqual(["Alice"]);
    expect(rankItems(items, "ali", (s) => s)).toEqual(["Alice", "Alicia"]);
  });

  it("respects cap parameter", () => {
    const result = rankItems(items, "", (s) => s, 2);
    expect(result).toHaveLength(2);
  });

  it("caps with empty query", () => {
    const result = rankItems(items, "", (s) => s, 3);
    expect(result).toEqual(["Alice", "Bob", "Charlie"]);
  });

  it("returns empty when no match", () => {
    expect(rankItems(items, "zzz", (s) => s)).toEqual([]);
  });

  it("works with object items and accessor", () => {
    const speakers = [
      { id: "1", name: "Alice" },
      { id: "2", name: "Bob" },
    ];
    const result = rankItems(speakers, "ali", (s) => s.name);
    expect(result).toEqual([{ id: "1", name: "Alice" }]);
  });
});

describe("moveActive", () => {
  it("returns -1 when count is 0", () => {
    expect(moveActive(0, "ArrowDown", 0)).toBe(-1);
  });

  it("moves down", () => {
    expect(moveActive(0, "ArrowDown", 5)).toBe(1);
  });

  it("clamps at the end", () => {
    expect(moveActive(4, "ArrowDown", 5)).toBe(4);
  });

  it("moves up", () => {
    expect(moveActive(3, "ArrowUp", 5)).toBe(2);
  });

  it("clamps at the start", () => {
    expect(moveActive(0, "ArrowUp", 5)).toBe(0);
  });

  it("jumps to start on Home", () => {
    expect(moveActive(3, "Home", 5)).toBe(0);
  });

  it("jumps to end on End", () => {
    expect(moveActive(1, "End", 5)).toBe(4);
  });

  it("returns index for unknown keys", () => {
    expect(moveActive(2, "x", 5)).toBe(2);
  });
});
