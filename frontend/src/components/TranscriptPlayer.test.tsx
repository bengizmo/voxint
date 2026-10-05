// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useLayoutEffect, useRef } from "react";
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

it("preserves astral text, overlapping highlights and nested selection offsets", async () => {
  const { selectionToCapture } = await import("../lib/selection");
  const text = "😀 um hello.";
  const units = [{ start: 1, end: 2, from: 2, to: 4, removed: "filler" as const, protected: false, mark: null }];
  const { container } = render(<TranscriptPlayer
    runId="run" mediaUrl="/audio" segments={[{ ...twoLines[0], text }]}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 1 }} lowConfidenceThreshold={0.5}
    annotationSpans={new Map([[0, [{ start: 3, end: 8, colorIndex: 1 }]]])}
    wordMarks={new Map([[0, { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units }]])}
  />);
  const wrapper = container.querySelector<HTMLElement>("[data-seg-text]")!;
  expect(wrapper.textContent).toBe(text);
  const nested = wrapper.querySelector(".wm-removed mark.hl-1")!;
  expect(nested.textContent).toBe("m");
  expect(nested.parentElement?.title).toBe("removed by filler clean-up");
  const range = document.createRange();
  range.setStart(nested.firstChild!, 0);
  range.setEnd(nested.firstChild!, 1);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  expect(selectionToCapture(container)).toMatchObject({ start: { offset: 3 }, end: { offset: 4 }, clientQuote: "m" });
  // Element-boundary selection also accounts for all preceding nested nodes.
  range.selectNodeContents(nested.parentElement!);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  expect(selectionToCapture(container)).toMatchObject({ start: { offset: 3 }, end: { offset: 4 } });
  window.getSelection()!.removeAllRanges();
});

it("renders focusable cleanup units and plain gaps without changing text", () => {
  const units = [
    { start: 0, end: 1, from: 0, to: 1, removed: null, protected: false, mark: null },
    { start: 1, end: 2, from: 2, to: 4, removed: null, protected: true, mark: "keep" as const },
    { start: 2, end: 3, from: 5, to: 11, removed: "omit" as const, protected: false, mark: "omit" as const },
  ];
  const selected = vi.fn();
  const { container } = render(<TranscriptPlayer
    runId="run" mediaUrl="/audio" segments={[{ ...twoLines[0], text: "😀 um hello." }]}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 1 }} lowConfidenceThreshold={0.5}
    cleanupIndex={0} cleanupFocusKey="seg-0" cursorIndex={0} onFocusUnit={selected}
    wordMarks={new Map([[0, { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units }]])}
  />);
  const wrapper = container.querySelector("[data-seg-text]")!;
  expect(wrapper.textContent).toBe("😀 um hello.");
  const buttons = wrapper.querySelectorAll<HTMLButtonElement>("button");
  expect(buttons).toHaveLength(3);
  expect([...buttons].every((b) => b.tabIndex === 0)).toBe(true);
  expect(document.activeElement).toBe(buttons[0]);
  fireEvent.keyDown(buttons[0], { key: "ArrowRight" });
  expect(document.activeElement).toBe(buttons[1]);
  expect(selected).toHaveBeenLastCalledWith(units[1]);
  expect(buttons[1].className).toContain("wm-keep");
  expect(buttons[2].className).toContain("wm-omit");
  fireEvent.keyDown(buttons[1], { key: "ArrowLeft" });
  expect(document.activeElement).toBe(buttons[0]);
});

it("does not overwrite a unit selected between cleanup mounting and passive effects", () => {
  const units = [
    { start: 0, end: 1, from: 0, to: 1, removed: null, protected: false, mark: null },
    { start: 1, end: 2, from: 2, to: 4, removed: "filler" as const, protected: false, mark: null },
    { start: 2, end: 3, from: 5, to: 11, removed: null, protected: false, mark: null },
  ];
  const selected = vi.fn();
  function EarlySelection() {
    const root = useRef<HTMLDivElement>(null);
    // Deterministically deliver a selection after the child commits but before
    // passive effects, the ordering that previously reset an operator's choice.
    useLayoutEffect(() => {
      root.current?.querySelector<HTMLButtonElement>('[data-word-unit="2"]')?.focus();
    }, []);
    return <div ref={root}><TranscriptPlayer
      runId="run" mediaUrl="/audio" segments={[{ ...twoLines[0], text: "😀 um hello." }]}
      capability={{ seekEnabled: true, reasons: [], mediaDuration: 1 }} lowConfidenceThreshold={0.5}
      cleanupIndex={0} cleanupFocusKey="seg-0" cursorIndex={0} onFocusUnit={selected}
      wordMarks={new Map([[0, { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units }]])}
    /></div>;
  }
  render(<EarlySelection />);
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "hello." }));
  expect(selected).toHaveBeenLastCalledWith(units[2]);
});

it("leaves ordinary units in one text node outside clean-up mode", () => {
  const { container } = render(<TranscriptPlayer
    runId="run" mediaUrl="/audio" segments={[{ ...twoLines[0], text: "hello there" }]}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 1 }} lowConfidenceThreshold={0.5}
    wordMarks={new Map([[0, { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units: [
      { start: 0, end: 1, from: 0, to: 5, removed: null, protected: false, mark: null },
      { start: 1, end: 2, from: 6, to: 11, removed: null, protected: false, mark: null },
    ] }]])}
  />);
  const text = container.querySelector("[data-seg-text]")!;
  expect(text.textContent).toBe("hello there");
  expect(text.childNodes).toHaveLength(1);
  expect(text.firstChild?.nodeType).toBe(Node.TEXT_NODE);
});

it("consumes word focus once per entry/cursor request even when units remount", () => {
  const props = {
    runId: "run", mediaUrl: "/audio", segments: [{ ...twoLines[0], text: "um hello" }],
    capability: { seekEnabled: true, reasons: [], mediaDuration: 1 }, lowConfidenceThreshold: 0.5,
    cursorIndex: 0, cleanupFocusKey: "seg-0",
    wordMarks: new Map([[0, { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units: [
      { start: 0, end: 1, from: 0, to: 2, removed: "filler" as const, protected: false, mark: null },
      { start: 1, end: 2, from: 3, to: 8, removed: null, protected: false, mark: null },
    ] }]]),
  };
  const view = render(<TranscriptPlayer {...props} cleanupIndex={0} />);
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "um: removed by filler clean-up" }));
  const hello = screen.getByRole("button", { name: "hello" });
  act(() => hello.focus());
  view.rerender(<TranscriptPlayer {...props} wordMarks={new Map(props.wordMarks)} cleanupIndex={0} />);
  expect(document.activeElement).toBe(hello);
  view.rerender(<TranscriptPlayer {...props} cleanupIndex={null} />);
  const control = screen.getByRole("button", { name: "Play line at 0:00.00" });
  act(() => control.focus());
  view.rerender(<TranscriptPlayer {...props} cleanupIndex={0} />);
  expect(document.activeElement).toBe(control);
  // Leaving and entering is a new deliberate request.
  view.rerender(<TranscriptPlayer {...props} cleanupFocusKey={null} cleanupIndex={null} />);
  view.rerender(<TranscriptPlayer {...props} cleanupIndex={0} />);
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "um: removed by filler clean-up" }));
});
