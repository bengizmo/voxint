// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { apiFetch } from "../lib/api-client";
import { SpeakerRail, type LabelsResult } from "./SpeakerRail";

vi.mock("../lib/api-client", async (original) => ({
  ...await original<typeof import("../lib/api-client")>(),
  apiFetch: vi.fn(),
}));

Element.prototype.scrollIntoView ??= () => {};
beforeEach(() => {
  vi.mocked(apiFetch).mockReset();
  vi.stubGlobal("CSS", { escape: (s: string) => s.replaceAll(":", "\\:") });
  vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const labels = ["SPEAKER_00", "SPEAKER_01"].map((label) => ({
  label, paletteIndex: 0, turnCount: 2, totalSeconds: 10,
  resolution: "unresolved", speakerId: null, speakerName: null,
  cosineConfidence: null, cosineSpeakerId: null, cosineSpeakerName: null,
  cosineGrounded: false, llmHintName: null, band: null, bandReason: null,
  candidatePromptAllowed: false, candidateSpeakerId: null, candidateSpeakerName: null,
  matchDecision: null, matchReason: null, matchSimilarity: null,
  matchMargin: null, matchVoteAgreement: null, matchEligibleSeconds: 0,
}));
const data: LabelsResult = {
  labels, segments: [], progress: { verified: 0, total: 2 },
};
function jsonResponse(value: unknown): Response {
  return { json: async () => value } as Response;
}
function setup() {
  const writeGuard = { busy: false, busyRef: { current: false }, setBusy: vi.fn() };
  const props: ComponentProps<typeof SpeakerRail> = {
    runId: "run", reviewToken: "token", writable: true, labelStates: labels,
    speakers: [{ id: "alice", displayName: "Alice" }],
    onClaimLost: vi.fn(), onLabelsChanged: vi.fn(), writeGuard,
  };
  return { ...render(<SpeakerRail {...props} />), props, writeGuard };
}
function createSpeaker() {
  fireEvent.click(screen.getByRole("button", { name: "SPEAKER_00: choose who this is" }));
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "Bob" } });
  fireEvent.click(screen.getByRole("option", { name: 'Create "Bob"' }));
}
async function previewMerge() {
  fireEvent.click(screen.getByText("Same speaker across labels?"));
  fireEvent.click(screen.getByRole("checkbox", { name: /SPEAKER_00/ }));
  fireEvent.click(screen.getByRole("checkbox", { name: /SPEAKER_01/ }));
  fireEvent.click(screen.getByRole("button", { name: "Merge target speaker" }));
  fireEvent.click(screen.getByRole("option", { name: "Alice" }));
  vi.mocked(apiFetch).mockResolvedValueOnce(jsonResponse({
    labels: labels.map((s) => s.label), speakerId: "alice", speakerName: "Alice",
    turnsMoved: 2, expected: { SPEAKER_00: null, SPEAKER_01: null },
  }));
  fireEvent.click(screen.getByRole("button", { name: "Preview merge…" }));
  return screen.findByRole("button", { name: "Confirm merge" });
}

it.each(["decide", "enroll"])("%s no-ops while the shared ref is held", async (action) => {
  const { writeGuard } = setup();
  writeGuard.busyRef.current = true;
  if (action === "decide") fireEvent.click(screen.getAllByRole("button", { name: "Can't tell" })[0]);
  else {
    createSpeaker();
    await act(async () => {});
  }
  expect(apiFetch).not.toHaveBeenCalled();
  expect(writeGuard.busyRef.current).toBe(true);
  expect(writeGuard.setBusy).not.toHaveBeenCalled();
});

it.each(["decide", "enroll"])("%s holds the shared guard until its response is adopted", async (action) => {
  let respond!: (response: Response) => void;
  vi.mocked(apiFetch).mockReturnValueOnce(new Promise<Response>((resolve) => { respond = resolve; }));
  const { props, writeGuard } = setup();
  if (action === "decide") fireEvent.click(screen.getAllByRole("button", { name: "Can't tell" })[0]);
  else createSpeaker();
  expect(apiFetch).toHaveBeenCalledExactlyOnceWith(
    `/review/run/labels/SPEAKER_00/${action === "decide" ? "decision" : "enroll"}`,
    expect.objectContaining({ method: "POST" }),
  );
  expect(writeGuard.busyRef.current).toBe(true);
  expect(writeGuard.setBusy).toHaveBeenCalledExactlyOnceWith(true);
  expect(props.onLabelsChanged).not.toHaveBeenCalled();
  await act(async () => respond(jsonResponse(data)));
  expect(props.onLabelsChanged).toHaveBeenCalledExactlyOnceWith(data);
  expect(writeGuard.busyRef.current).toBe(false);
  expect(writeGuard.setBusy).toHaveBeenLastCalledWith(false);
});

it("leaves preview unguarded and blocks confirm merge while the shared ref is held", async () => {
  const { writeGuard } = setup();
  writeGuard.busyRef.current = true;
  const confirm = await previewMerge();
  expect(apiFetch).toHaveBeenCalledTimes(1);
  fireEvent.click(confirm);
  expect(apiFetch).toHaveBeenCalledTimes(1);
  expect(writeGuard.busyRef.current).toBe(true);
  expect(writeGuard.setBusy).not.toHaveBeenCalled();
});

it("holds the shared guard while confirm merge is in flight", async () => {
  const { props, writeGuard } = setup();
  const confirm = await previewMerge();
  let respond!: (response: Response) => void;
  vi.mocked(apiFetch).mockReturnValueOnce(new Promise<Response>((resolve) => { respond = resolve; }));
  act(() => { confirm.click(); confirm.click(); });
  expect(apiFetch).toHaveBeenCalledTimes(2);
  expect(vi.mocked(apiFetch).mock.calls[1][0]).toBe("/review/run/merge");
  expect(writeGuard.busyRef.current).toBe(true);
  expect(writeGuard.setBusy).toHaveBeenCalledExactlyOnceWith(true);
  expect(props.onLabelsChanged).not.toHaveBeenCalled();
  await act(async () => respond(jsonResponse(data)));
  expect(props.onLabelsChanged).toHaveBeenCalledExactlyOnceWith(data);
  expect(writeGuard.busyRef.current).toBe(false);
  expect(writeGuard.setBusy).toHaveBeenLastCalledWith(false);
});

it("disables ruling and merge controls while the editor is busy", async () => {
  const { props, rerender, writeGuard } = setup();
  await previewMerge();
  rerender(<SpeakerRail {...props} writeGuard={{ ...writeGuard, busy: true }} />);
  await waitFor(() => {
    for (const button of screen.getAllByRole("button", { name: "Can't tell" })) {
      expect(button).toHaveProperty("disabled", true);
    }
    expect(screen.getByRole("button", { name: "SPEAKER_00: choose who this is" })).toHaveProperty("disabled", true);
    expect(screen.getByRole("button", { name: "Confirm merge" })).toHaveProperty("disabled", true);
  });
});
