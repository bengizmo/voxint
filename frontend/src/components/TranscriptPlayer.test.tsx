// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TranscriptPlayer, type Segment } from "./TranscriptPlayer";

const splitPart = {
  start: 0, end: 1, speaker: "Alice", label: "S0", paletteIndex: 0,
  confidence: null, segmentId: "seg-0", sourceSegmentId: "seg-0",
  text: "First part", reviewTarget: true, verified: false, corrected: false,
  wordStart: 0, wordEnd: 2, wordRangeSpeakerId: null, corrections: null,
} satisfies Segment;

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// #717: an opacity below 1 on any element between the line and the open list
// makes a stacking context, which traps the list's z-index under the next
// transcript line and the sticky segment-actions bar. jsdom cannot paint, so
// pin the cause: no opacity utility on the list's ancestors inside the line.
it("keeps the split-part speaker picker out of an opacity stacking context", () => {
  render(<TranscriptPlayer
    runId="run" mediaUrl="/audio" segments={[splitPart]}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 1 }}
    lowConfidenceThreshold={0.5}
    reassignSpeakers={[{ id: "alice", displayName: "Alice" }]}
    onReassign={() => {}}
  />);

  const picker = screen.getByRole("button", { name: 'Reassign speaker for "First part"' });
  const line = picker.closest("p.tp-line");
  expect(line).not.toBeNull();
  for (let el: Element | null = picker; el && el !== line; el = el.parentElement) {
    expect(el.className).not.toMatch(/(^|\s)opacity-/);
  }
});
