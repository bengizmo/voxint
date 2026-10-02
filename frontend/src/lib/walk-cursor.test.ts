// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import type { Segment } from "../components/TranscriptPlayer";
import { useWalkCursor } from "./editor-mutations";

const segments: Segment[] = [0, 1, 2].map((index) => ({
  start: index, end: index + 1, speaker: "Alice", label: "A",
  segmentId: `seg-${index}`, sourceSegmentId: `seg-${index}`, text: `Text ${index}`,
  reviewTarget: index !== 1, verified: false, corrected: false,
  wordStart: null, wordEnd: null, wordRangeSpeakerId: null,
  paletteIndex: null, confidence: null, corrections: null,
}));

function walk(canLeave: () => boolean) {
  const play = vi.fn();
  const hook = renderHook(() => useWalkCursor(segments, segments, play, canLeave));
  return { play, hook };
}

it("keeps the cursor and plays nothing when canLeave refuses", () => {
  const { play, hook } = walk(() => false);
  let went: boolean[] = [];
  act(() => {
    went = [
      hook.result.current.goTo(2),
      hook.result.current.select(2),
      hook.result.current.jumpNext(),
    ];
  });
  expect(went).toEqual([false, false, false]);
  expect(hook.result.current.cursor).toBe(0);
  expect(play).not.toHaveBeenCalled();
});

it("moves when canLeave allows; select moves without playing", () => {
  const { play, hook } = walk(() => true);
  act(() => { expect(hook.result.current.select(1)).toBe(true); });
  expect(hook.result.current.cursor).toBe(1);
  expect(play).not.toHaveBeenCalled();

  // Line 1 is not a review target; from it, the next target is line 2.
  act(() => { expect(hook.result.current.jumpNext()).toBe(true); });
  expect(hook.result.current.cursor).toBe(2);
  expect(play).toHaveBeenCalledWith(2);
});

it("never asks canLeave for the line the cursor is on", () => {
  const canLeave = vi.fn(() => false);
  const { play, hook } = walk(canLeave);
  act(() => { expect(hook.result.current.goTo(0)).toBe(true); });
  expect(canLeave).not.toHaveBeenCalled();
  expect(play).toHaveBeenCalledWith(0);
});

it("refuses a negative index without asking", () => {
  const canLeave = vi.fn(() => true);
  const { hook } = walk(canLeave);
  act(() => { expect(hook.result.current.goTo(-1)).toBe(false); });
  expect(canLeave).not.toHaveBeenCalled();
});
