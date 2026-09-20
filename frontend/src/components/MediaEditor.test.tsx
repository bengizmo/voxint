// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MediaEditor, type MediaEditorProps } from "./MediaEditor";
import { apiFetch } from "../lib/api-client";
import type { Segment, TranscriptPlayerProps } from "./TranscriptPlayer";

vi.mock("../lib/api-client", async (original) => ({
  ...await original<typeof import("../lib/api-client")>(),
  apiFetch: vi.fn(),
}));
vi.mock("./AnnotationLayer", () => ({
  useAnnotations: () => ({ reload: vi.fn(), toolbar: null, panel: null }),
}));
vi.mock("./SpeakerRail", () => ({ SpeakerRail: () => null }));
vi.mock("./OutlinePanel", () => ({ OutlinePanel: () => null }));
vi.mock("./KeymapHelp", () => ({ KeymapHelp: () => null }));
vi.mock("./TranscriptPlayer", () => ({
  TranscriptPlayer: (props: TranscriptPlayerProps) => (
    <div>{props.segments.map((seg, index) => (
      <button key={index} onClick={(event) => {
        event.currentTarget.focus();
        props.onSpeakerClick?.(index, new DOMRect(0, 0, 100, 20));
      }}>Speaker {index}: {seg.speaker}</button>
    ))}</div>
  ),
}));

const segments = [0, 1].map((index) => ({
  start: index, end: index + 1, speaker: "Alice",
  label: "VOICE/A", segmentId: `seg-${index}`, sourceSegmentId: `seg-${index}`,
  text: `Text ${index}`, reviewTarget: true, verified: false, corrected: false,
  wordStart: null, wordEnd: null, wordRangeSpeakerId: null,
  paletteIndex: null, confidence: null, corrections: null, rawText: null,
})) satisfies Segment[];

const defaultLabelStates = [{
  label: "VOICE/A", paletteIndex: 0, turnCount: 2, totalSeconds: 2,
  resolution: "human_assign", speakerId: "alice", speakerName: "Alice",
  cosineConfidence: null, cosineSpeakerId: null, cosineSpeakerName: null,
  cosineGrounded: false, llmHintName: null, band: null, bandReason: null,
  candidatePromptAllowed: false, candidateSpeakerId: null, candidateSpeakerName: null,
  matchDecision: null, matchReason: null, matchSimilarity: null,
  matchMargin: null, matchVoteAgreement: null, matchEligibleSeconds: 0,
}];

function setup(overrides: Partial<MediaEditorProps> = {}) {
  render(<MediaEditor
    mediaId="media" runId="run" mediaUrl="/audio"
    segments={segments}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }}
    lowConfidenceThreshold={0.5}
    reviewToken="claim"
    initialProgress={{ verified: 0, total: 2 }}
    speakers={[{ id: "alice", displayName: "Alice" }, { id: "bob", displayName: "Bob" }]}
    renameCsrf="rename-token"
    labelStates={defaultLabelStates}
    {...overrides}
  />);
  fireEvent.click(screen.getByRole("button", { name: "Speaker 1: Alice" }));
}
function body() {
  return new URLSearchParams(String(vi.mocked(apiFetch).mock.calls[0][1]?.body));
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("CSS", { escape: (s: string) => s.replaceAll(":", "\\:") });
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(apiFetch).mockResolvedValue({
    json: async () => ({ segments, progress: { verified: 0, total: 2 }, labels: [] }),
  } as unknown as Response);
});
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

it("assigns the clicked segment rather than the previous cursor", async () => {
  setup();
  fireEvent.click(screen.getByRole("radio", { name: "Just this segment" }));
  fireEvent.click(screen.getByRole("option", { name: "Bob" }));
  await waitFor(() => expect(apiFetch).toHaveBeenCalledOnce());
  expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe("/review/run/segments/seg-1/relabel");
  expect(body().get("speaker_id")).toBe("bob");
  expect(body().get("token")).toBe("claim");
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("encodes label-scope assignment paths", async () => {
  setup();
  fireEvent.click(screen.getByRole("radio", { name: /All segments/ }));
  fireEvent.click(screen.getByRole("option", { name: "Bob" }));
  await waitFor(() => expect(apiFetch).toHaveBeenCalledOnce());
  expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe("/review/run/labels/VOICE%2FA/decision");
  expect(body().get("action")).toBe("assign");
  expect(await screen.findByText("Assigned 2 segments to Bob.")).toBeTruthy();
});

it("enrolls at label scope and hides Create in segment scope", async () => {
  setup({ labelStates: [{
    ...defaultLabelStates[0], resolution: "unresolved", speakerId: null, speakerName: null,
  }] });
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "Mara" } });
  fireEvent.click(screen.getByRole("option", { name: 'Create "Mara"' }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe("/review/run/labels/VOICE%2FA/enroll");
  expect(body().get("display_name")).toBe("Mara");
});

it("suppresses Create option when segment scope is selected", () => {
  setup({ labelStates: [{
    ...defaultLabelStates[0], resolution: "unresolved", speakerId: null, speakerName: null,
  }] });
  fireEvent.click(screen.getByRole("radio", { name: "Just this segment" }));
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "Mara" } });
  expect(screen.queryByRole("option", { name: 'Create "Mara"' })).toBeNull();
});

it("sends rename CSRF and adopts the server's normalized name", async () => {
  vi.mocked(apiFetch).mockResolvedValue({
    json: async () => ({ id: "alice", displayName: "Alicia" }),
  } as Response);
  setup({ segments: segments.map((seg, index) =>
    index === 0 ? { ...seg, label: "VOICE/B" } : seg) });
  fireEvent.click(screen.getByRole("button", { name: /Rename/ }));
  fireEvent.change(screen.getByLabelText("Rename “Alice”"), { target: { value: " Alicia " } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByRole("button", { name: "Speaker 0: Alicia" });
  expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe("/speakers/alice/rename");
  expect(body().get("csrf_token")).toBe("rename-token");
  expect(screen.getByText("Renamed speaker to Alicia.")).toBeTruthy();
});

it("explains unsupported label reset without sending a mutation", () => {
  setup();
  fireEvent.click(screen.getByRole("radio", { name: /All segments/ }));
  fireEvent.click(screen.getByRole("button", { name: /Reset to detected speaker/ }));
  expect(apiFetch).not.toHaveBeenCalled();
  expect(screen.getByRole("alert").textContent).toContain("only supported for just this segment");
});


it("blocks global assignment and review shortcuts while the popover is open", () => {
  setup();
  const panel = screen.getByRole("dialog");
  for (const key of ["1", "=", "v"]) fireEvent.keyDown(panel, { key });
  expect(apiFetch).not.toHaveBeenCalled();
  expect(screen.getByRole("dialog")).toBe(panel);
});

it("explains copying the previous speaker at the first segment", () => {
  setup();
  fireEvent.keyDown(screen.getByRole("combobox"), { key: "Escape" });
  fireEvent.keyDown(document.body, { key: "=" });
  expect(screen.getByText("No previous segment to copy from.")).toBeTruthy();
  expect(apiFetch).not.toHaveBeenCalled();
});
