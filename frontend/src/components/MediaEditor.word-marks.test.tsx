// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MediaEditor } from "./MediaEditor";
import type { Segment } from "./TranscriptPlayer";
import { ApiError, apiFetch } from "../lib/api-client";
import type { WordMarksPayload, WordMarkUnit } from "../lib/word-marks";
import type { LabelsResult } from "./SpeakerRail";

vi.mock("../lib/api-client", async (original) => ({
  ...await original<typeof import("../lib/api-client")>(), apiFetch: vi.fn(),
}));
vi.mock("./AnnotationLayer", () => ({ useAnnotations: () => ({ reload: vi.fn(), toolbar: null, panel: null }) }));
const rail = vi.hoisted(() => ({ adopt: null as ((result: LabelsResult) => void) | null }));
const busyScheduling = vi.hoisted(() => ({ holdState: false, sawPendingWithIdleState: false }));
vi.mock("../lib/editor-mutations", async (original) => {
  const actual = await original<typeof import("../lib/editor-mutations")>();
  return { ...actual, useBusyGuard: () => {
    const guard = actual.useBusyGuard();
    if (busyScheduling.holdState && guard.busyRef.current && !guard.busy) {
      busyScheduling.sawPendingWithIdleState = true;
    }
    // Model the browser render where cursor updates have landed but busy
    // state has not. Settlement must work even with no busy-state flip.
    return { ...guard, setBusy: (busy: boolean) => {
      if (!busyScheduling.holdState) guard.setBusy(busy);
    } };
  } };
});
vi.mock("./SpeakerRail", () => ({ SpeakerRail: (props: { onLabelsChanged: (result: LabelsResult) => void }) => {
  rail.adopt = props.onLabelsChanged; return null;
} }));
vi.mock("./OutlinePanel", () => ({ OutlinePanel: () => null }));

const segments: Segment[] = ["😀 um hello.", "uh there"].map((text, i) => ({
  text, start: i, end: i + 1, speaker: "S0", label: "S0", paletteIndex: 0, confidence: null,
  segmentId: `seg-${i}`, sourceSegmentId: `seg-${i}`, reviewTarget: true, verified: false, corrected: false,
  wordStart: null, wordEnd: null, wordRangeSpeakerId: null, corrections: null,
}));
const units: WordMarkUnit[] = [
  { start: 0, end: 1, from: 0, to: 1, removed: null, protected: false, mark: null },
  { start: 1, end: 2, from: 2, to: 4, removed: "filler", protected: false, mark: null },
  { start: 2, end: 3, from: 5, to: 11, removed: null, protected: false, mark: null },
];
let state: WordMarksPayload;
let version: number;
const response = (data: unknown) => ({ json: async () => data }) as Response;
const writes = () => vi.mocked(apiFetch).mock.calls.filter(([, init]) => init?.method === "POST");
const focusedGets = () => vi.mocked(apiFetch).mock.calls.filter(([url, init]) =>
  url.includes("/word-marks?segment=") && init?.method !== "POST").map(([url]) => url);
function deferred() {
  let resolve!: (value: Response) => void;
  const promise = new Promise<Response>((done) => { resolve = done; });
  return { promise, resolve };
}
async function readyAction(name: string) {
  const button = screen.getByRole<HTMLButtonElement>("button", { name });
  await waitFor(() => expect(button.disabled).toBe(false));
  return button;
}
function setup(reviewToken: string | null = "claim") {
  return render(<MediaEditor mediaId="media" runId="run" mediaUrl="/audio" segments={segments}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }} lowConfidenceThreshold={0.5}
    reviewToken={reviewToken} multiUser initialProgress={{ verified: 0, total: 2 }} speakers={[]} claimCsrf="csrf" />);
}
function key(key: string, target: HTMLElement | Window = window) { fireEvent.keyDown(target, { key }); }
async function enter() {
  key("c");
  const um = await screen.findByRole("button", { name: "um: removed by filler clean-up" });
  // Finding the DOM alone did not guarantee the mount focus/selection had
  // settled. Subsequent actions depend on that actual state transition.
  await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "😀" })));
  await readyAction("Omit o");
  return um;
}
beforeEach(() => {
  busyScheduling.holdState = false;
  busyScheduling.sawPendingWithIdleState = false;
  version = 0;
  state = { runId: "run", version: "0.hash", fillerListDefault: true, detectionError: null, stale: [], emissions: [
    { segmentId: "seg-0", wordStart: null, wordEnd: null, markable: true, reason: null, units: units.map((u) => ({ ...u })) },
    { segmentId: "seg-1", wordStart: null, wordEnd: null, markable: true, reason: null, units: [
      { ...units[1], start: 0, end: 1, from: 0, to: 2 }, { ...units[2], start: 1, end: 2, from: 3, to: 8 },
    ] },
  ] };
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  Element.prototype.scrollIntoView ??= () => {};
  vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.mocked(apiFetch).mockReset().mockImplementation(async (url, init) => {
    if (url.includes("/word-marks")) {
      if (init?.method === "POST") {
        const body = new URLSearchParams(String(init.body));
        const segId = url.split("/segments/")[1].split("/")[0];
        const unit = state.emissions.find((e) => e.segmentId === segId)?.units.find((u) => u.start === Number(body.get("start")));
        const action = body.get("action") as "keep" | "omit" | "clear";
        if (unit) { unit.mark = action === "clear" ? null : action; unit.protected = action === "keep"; }
        state = { ...state, version: `${++version}.hash`, stale: [] };
        return response({ ...structuredClone(state), undo: { kind: "word-mark", markId: `mark-${version}`, expiresAt: new Date(Date.now() + 60_000).toISOString() } });
      }
      const focus = new URL(url, "http://localhost").searchParams.get("segment");
      return response({ ...structuredClone(state), emissions: state.emissions.map((e) => ({ ...e,
        units: e.segmentId === focus ? structuredClone(e.units) : structuredClone(e.units.filter((u) => u.removed || u.mark)),
      })) });
    }
    if (url.endsWith("/undo/word-mark")) {
      state.emissions[0].units[1] = { ...units[1] };
      state = { ...state, version: `${++version}.hash` };
      return response(structuredClone(state));
    }
    if (url.endsWith("/words")) return response({ splittable: true, reason: null, words: [{ word: "😀", start: 0, end: 0.3 }, { word: " um", start: 0.3, end: 0.6 }] });
    if (url.endsWith("/text")) return response({ text: "Edited words.", corrected: true, verified: false, marksCleared: 2, progress: { verified: 0, total: 2 } });
    if (url.endsWith("/verify")) return response({ text: segments[0].text, corrected: false, verified: true, progress: { verified: 1, total: 2 } });
    return response({ speakerIds: [] });
  });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("fetches mount detection, toggles c/f/o, updates styles directly and undoes", async () => {
  const { container } = setup();
  await waitFor(() => expect(container.querySelector(".wm-removed")?.textContent).toBe("um"));
  const um = await enter();
  act(() => um.focus()); key("f", um);
  await screen.findByRole("button", { name: "um: kept" });
  expect(writes()[0][0]).toBe("/review/run/segments/seg-0/word-marks");
  expect(Object.fromEntries(new URLSearchParams(String(writes()[0][1]?.body)))).toMatchObject({ token: "claim", action: "keep", start: "1", end: "2" });
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  const restored = await screen.findByRole("button", { name: "um: removed by filler clean-up" });
  await readyAction("Omit o");
  key("ArrowRight", restored);
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "hello." }));
  key("o", document.activeElement as HTMLElement);
  const omitted = await screen.findByRole("button", { name: "hello.: omitted" });
  expect(omitted.className).toContain("wm-omit");
  await readyAction("Omit o");
  key("o", omitted);
  await screen.findByRole("button", { name: "hello." });
  key("Escape");
  expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull();
  expect(container.querySelector("[data-seg-text]")?.textContent).toBe(segments[0].text);
  const before = writes().length; key("f"); key("o");
  expect(writes()).toHaveLength(before);
});

it("keeps navigation and verify active from unit buttons and refreshes focus parents", async () => {
  setup();
  const um = await enter();
  expect(focusedGets()).toEqual(["/review/run/word-marks?segment=seg-0"]);
  key("j", um);
  await screen.findByRole("button", { name: "uh: removed by filler clean-up" });
  expect(focusedGets()).toEqual([
    "/review/run/word-marks?segment=seg-0", "/review/run/word-marks?segment=seg-1",
  ]);
  key("k", document.activeElement as HTMLElement);
  const current = await screen.findByRole("button", { name: "um: removed by filler clean-up" });
  expect(focusedGets()).toEqual([
    "/review/run/word-marks?segment=seg-0", "/review/run/word-marks?segment=seg-1",
    "/review/run/word-marks?segment=seg-0",
  ]);
  key("v", current);
  await waitFor(() => expect(writes().some(([url]) => url.endsWith("/verify"))).toBe(true));
});

it.each(["mark", "undo"])("fetches the new cursor's marks once when a %s finishes during navigation", async (action) => {
  setup();
  const um = await enter();
  if (action === "undo") {
    act(() => um.focus());
    key("f", um);
    await screen.findByRole("button", { name: "um: kept" });
    await readyAction("Keep f");
  }
  const fetch = vi.mocked(apiFetch).getMockImplementation()!;
  const nextRead = deferred();
  let nextPayload!: Promise<Response>;
  let finishWrite!: () => void;
  vi.mocked(apiFetch).mockImplementation((url, init) => {
    if (init?.method === "POST" && (url.endsWith("/word-marks") || url.endsWith("/undo/word-mark"))) {
      return new Promise<Response>((resolve) => {
        finishWrite = () => { resolve(fetch(url, init)); };
      });
    }
    if (url === "/review/run/word-marks?segment=seg-1") {
      nextPayload = fetch(url, init);
      return nextRead.promise;
    }
    return fetch(url, init);
  });
  if (action === "undo") fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  else {
    act(() => um.focus());
    key("f", um);
  }
  key("j");
  await screen.findByDisplayValue(segments[1].text);
  expect(focusedGets()).toEqual(["/review/run/word-marks?segment=seg-0"]);
  await act(async () => { finishWrite(); });
  await waitFor(() => expect(focusedGets()).toEqual([
    "/review/run/word-marks?segment=seg-0", "/review/run/word-marks?segment=seg-1",
  ]));
  expect(screen.queryByRole("button", { name: "uh: removed by filler clean-up" })).toBeNull();
  await act(async () => { nextRead.resolve(await nextPayload); });
  await screen.findByRole("button", { name: "uh: removed by filler clean-up" });
  await readyAction("Omit o");
  expect(vi.mocked(apiFetch).mock.calls.filter(([url, init]) =>
    url === "/review/run/word-marks?segment=seg-1" && init?.method !== "POST")).toHaveLength(1);
});

it.each([
  { action: "mark", saved: true }, { action: "undo", saved: true },
  { action: "mark", saved: false }, { action: "undo", saved: false },
])("defers the cursor GET through $action settlement (saved=$saved) even when no render sees busy=true", async ({ action, saved }) => {
  const { container } = setup();
  const um = await enter();
  act(() => um.focus());
  if (action === "undo") {
    key("f", um);
    await screen.findByRole("button", { name: "um: kept" });
    await readyAction("Keep f");
  }
  busyScheduling.holdState = true;
  const fetch = vi.mocked(apiFetch).getMockImplementation()!;
  const post = deferred();
  const read = deferred();
  let finishPost!: () => Promise<void>;
  let readPayload!: Promise<Response>;
  let postResolved = false;
  vi.mocked(apiFetch).mockImplementation((url, init) => {
    if (init?.method === "POST") {
      finishPost = async () => {
        const result = saved ? await fetch(url, init) : {
          json: async () => { throw new ApiError(500, "write failed"); },
        } as unknown as Response;
        postResolved = true;
        post.resolve(result);
      };
      return post.promise;
    }
    if (url === "/review/run/word-marks?segment=seg-1") {
      expect(postResolved).toBe(true);
      readPayload = fetch(url, init);
      return read.promise;
    }
    return fetch(url, init);
  });
  act(() => {
    if (action === "undo") fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    else key("f", um);
    key("j");
  });
  await screen.findByDisplayValue(segments[1].text);
  expect(busyScheduling.sawPendingWithIdleState).toBe(true);
  expect(focusedGets()).toEqual(["/review/run/word-marks?segment=seg-0"]);
  await act(async () => { await finishPost(); });
  await waitFor(() => expect(focusedGets()).toEqual([
    "/review/run/word-marks?segment=seg-0", "/review/run/word-marks?segment=seg-1",
  ]));
  // Before the follow-up GET lands, the completed write is already adopted
  // for its own segment rather than discarded as a superseded response.
  expect(container.querySelector('[data-seg-text] .wm-keep')?.textContent ?? null)
    .toBe((action === "mark" && saved) || (action === "undo" && !saved) ? "um" : null);
  if (!saved) expect(screen.getByText("write failed")).toBeTruthy();
  await act(async () => { read.resolve(await readPayload); });
  await screen.findByRole("button", { name: "uh: removed by filler clean-up" });
  expect(focusedGets()).toHaveLength(2);
});

it.each(["mark", "undo"])("preserves an adopted %s when a pre-write focused GET finishes late, without a release-time refetch", async (action) => {
  setup();
  const um = await enter();
  act(() => um.focus());
  if (action === "undo") {
    key("f", um);
    await screen.findByRole("button", { name: "um: kept" });
    await readyAction("Keep f");
  }
  const fetch = vi.mocked(apiFetch).getMockImplementation()!;
  const olderRead = deferred();
  const oldPayload = response(structuredClone(state));
  const write = deferred();
  let finishWrite!: () => Promise<void>;
  vi.mocked(apiFetch).mockImplementation((url, init) => {
    if (url === "/review/run/word-marks?segment=seg-0") return olderRead.promise;
    if (init?.method === "POST" && (url.endsWith("/word-marks") || url.endsWith("/undo/word-mark"))) {
      finishWrite = async () => { write.resolve(await fetch(url, init)); };
      return write.promise;
    }
    return fetch(url, init);
  });
  // Whole-run adoption refreshes marks at the existing focus. Keep its GET
  // pending while the next mutation invalidates it and adopts server truth.
  act(() => rail.adopt!({ segments, labels: [], progress: { verified: 0, total: 2 },
    undo: action === "undo" ? { kind: "word-mark", markId: "mark-1", expiresAt: new Date(Date.now() + 60_000).toISOString() } : undefined,
  }));
  expect(focusedGets()).toHaveLength(2);
  if (action === "undo") fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  else key("f", um);
  expect(screen.getByRole<HTMLButtonElement>("button", { name: "Keep f" }).disabled).toBe(true);
  await act(async () => { await finishWrite(); });
  const expectedName = action === "mark" ? "um: kept" : "um: removed by filler clean-up";
  const adopted = await screen.findByRole("button", { name: expectedName });
  await readyAction("Keep f");
  expect(document.activeElement).toBe(adopted);
  await act(async () => { olderRead.resolve(oldPayload); });
  expect(screen.getByRole("button", { name: expectedName })).toBe(adopted);
  expect(focusedGets()).toEqual([
    "/review/run/word-marks?segment=seg-0", "/review/run/word-marks?segment=seg-0",
  ]);
});

it("leaves split mode on entry, and split entry leaves cleanup; textarea suppresses keys", async () => {
  setup();
  fireEvent.click(screen.getByRole("button", { name: "Split" }));
  await screen.findByText("Split mode on — click a word to cut the segment before it.");
  await enter();
  expect(screen.getByRole("button", { name: "Split" }).getAttribute("aria-pressed")).toBe("false");
  const textarea = screen.getByRole("textbox", { name: "Corrected transcript text for this segment" });
  textarea.focus();
  for (const k of ["c", "f", "o", "v", "j", "k", "Escape"]) key(k, textarea);
  expect(screen.getByRole("region", { name: "Clean-up mode" })).toBeTruthy();
  expect(writes()).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Split" }));
  expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull();
});

it("Escape closes a modal before leaving cleanup", async () => {
  setup(); await enter(); key("?");
  const dialog = screen.getByRole("dialog");
  key("Escape", dialog);
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("region", { name: "Clean-up mode" })).toBeTruthy();
  key("Escape");
  expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull();
});

it("offers mouse actions, refuses ordinary keep, and clears selected marks", async () => {
  setup(); const um = await enter();
  key("ArrowRight", um);
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "hello." }));
  fireEvent.click(within(screen.getByRole("region", { name: "Clean-up mode" })).getByRole("button", { name: "Keep f" }));
  expect(await screen.findByText("Only a word the filler list removes can be kept.")).toBeTruthy();
  expect(writes()).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Omit o" }));
  await screen.findByRole("button", { name: "hello.: omitted" });
  fireEvent.click(await readyAction("Clear"));
  await screen.findByRole("button", { name: "hello." });
  expect(screen.getByText("Clean-up mark saved.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Done Escape" }));
  expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull();
});

it("shows marksCleared from a text save and refreshes after whole-run adoption", async () => {
  setup(); await screen.findByDisplayValue(segments[0].text);
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "Edited words." } });
  fireEvent.click(screen.getByRole("button", { name: /Save edit/ }));
  await screen.findByText("2 clean-up marks were cleared because the text changed.");
  const before = vi.mocked(apiFetch).mock.calls.filter(([url]) => url.includes("/word-marks")).length;
  act(() => rail.adopt!({ segments, labels: [], progress: { verified: 0, total: 2 } }));
  await waitFor(() => expect(vi.mocked(apiFetch).mock.calls.filter(([url]) => url.includes("/word-marks")).length).toBeGreaterThan(before));
});

it("clears the text-save marks notice when the cursor moves to another segment", async () => {
  setup();
  await screen.findByDisplayValue(segments[0].text);
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "Edited words." } });
  fireEvent.click(screen.getByRole("button", { name: /Save edit/ }));
  const notice = "2 clean-up marks were cleared because the text changed.";
  await screen.findByText(notice);
  key("j");
  await screen.findByDisplayValue(segments[1].text);
  expect(screen.queryByText(notice)).toBeNull();
  key("k");
  await screen.findByDisplayValue("Edited words.");
  expect(screen.queryByText(notice)).toBeNull();
});

it("clears a stale mark with its original parent/token range", async () => {
  state.stale = [{ segmentId: "seg-0", start: 4, end: 6, action: "omit", segmentStart: 0 }];
  setup();
  const notice = await screen.findByText("A clean-up mark no longer matches this text.");
  fireEvent.click(within(notice).getByRole("button", { name: "Clear" }));
  await waitFor(() => expect(screen.queryByText("A clean-up mark no longer matches this text.")).toBeNull());
  expect(Object.fromEntries(new URLSearchParams(String(writes()[0][1]?.body)))).toMatchObject({ action: "clear", start: "4", end: "6" });
});

it("explains unmarkable emissions and broken detection; requires a claim", async () => {
  state.detectionError = "Invalid saved list";
  state.emissions[0].markable = false; state.emissions[0].units = []; state.emissions[0].reason = "No recorded word timings.";
  setup(); key("c");
  await screen.findByText("No recorded word timings.");
  expect(screen.getByText(/Filler detection is unavailable/).textContent).toContain("Invalid saved list");
  cleanup(); setup(null); key("c");
  expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull();
});

it("distinguishes claim loss from ordinary mark conflicts", async () => {
  setup(); const um = await enter();
  vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError(409, "choose another word"));
  act(() => um.focus()); key("f", um);
  await screen.findByText("choose another word");
  expect(screen.getByRole("region", { name: "Clean-up mode" })).toBeTruthy();
  await readyAction("Keep f");
  vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError(409, "claim lost", "claim"));
  key("f", um);
  await waitFor(() => expect(screen.queryByRole("region", { name: "Clean-up mode" })).toBeNull());
});
