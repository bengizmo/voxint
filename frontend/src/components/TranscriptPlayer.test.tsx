// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TranscriptPlayer, type Segment } from "./TranscriptPlayer";

vi.mock("../lib/peaks", async (original) => ({
  ...await original<typeof import("../lib/peaks")>(),
  fetchPeaks: vi.fn(async () => ({
    version: 1, duration: 2, sampleRate: 16000, frameCount: 32000,
    samplesPerBucket: 160, peaks: [0.1, 0.2],
  })),
}));
// The strip's canvas needs layout jsdom lacks; a button stands in for a click
// on line 1's region.
vi.mock("./WaveformStrip", () => ({
  WaveformStrip: (props: { onRegionActivate: (index: number) => void }) => (
    <button onClick={() => props.onRegionActivate(1)}>Region 1</button>
  ),
}));

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

const twoLines = [0, 1].map((index) => ({
  ...splitPart, start: index, end: index + 1, segmentId: `seg-${index}`,
  sourceSegmentId: `seg-${index}`, text: `Line ${index}`, wordStart: null, wordEnd: null,
})) satisfies Segment[];

// #732: a region click the editor refuses (an unsaved edit) still plays the
// line but leaves the list where it is, so a repeat click lands on the strip.
it.each([
  ["moves the cursor", true, 1],
  ["keeps the cursor", false, 0],
])("plays a region click that %s, scrolling to it only then", async (_name, selects, scrolls) => {
  const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  Element.prototype.scrollIntoView ??= () => {};
  const scroll = vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
  const select = vi.fn(() => selects);
  render(<TranscriptPlayer
    runId="run" mediaUrl="/audio" segments={twoLines}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }}
    lowConfidenceThreshold={0.5}
    peaksUrl="/peaks"
    onSegmentSelect={select}
  />);
  const region = await screen.findByRole("button", { name: "Region 1" });
  scroll.mockClear();

  await act(async () => { fireEvent.click(region); });

  expect(select).toHaveBeenCalledWith(1);
  expect(play).toHaveBeenCalledOnce();
  expect(scroll).toHaveBeenCalledTimes(scrolls);
  play.mockRestore();
  scroll.mockRestore();
});

it("pausePlayback pauses the main audio and cancels its turn guard", async () => {
  const { createRef } = await import("react");
  const ref = createRef<import("./TranscriptPlayer").TranscriptPlayerHandle>();
  const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  const pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  const { container } = render(<TranscriptPlayer ref={ref}
    runId="run" mediaUrl="/audio" segments={twoLines}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }} lowConfidenceThreshold={0.5}
  />);
  act(() => { ref.current!.previewSegment(0); });
  act(() => { ref.current!.pausePlayback(); });
  expect(pause).toHaveBeenCalledOnce();
  const audio = container.querySelector("audio")!;
  audio.currentTime = 2;
  fireEvent.timeUpdate(audio);
  expect(pause).toHaveBeenCalledOnce();
  play.mockRestore();
  pause.mockRestore();
});
it("onMainPlay listens for main audio play and removes its listener on unmount", () => {
  const onMainPlay = vi.fn();
  const { container, unmount } = render(<TranscriptPlayer onMainPlay={onMainPlay}
    runId="run" mediaUrl="/audio" segments={twoLines}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }} lowConfidenceThreshold={0.5}
  />);
  const audio = container.querySelector("audio")!;
  fireEvent.play(audio);
  expect(onMainPlay).toHaveBeenCalledOnce();
  unmount();
  fireEvent.play(audio);
  expect(onMainPlay).toHaveBeenCalledOnce();
});
