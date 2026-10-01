// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MediaEditor, type MediaEditorProps } from "./MediaEditor";
import { apiFetch } from "../lib/api-client";
import type {
  Segment,
  TranscriptPlayerHandle,
  TranscriptPlayerProps,
} from "./TranscriptPlayer";

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
const player = vi.hoisted(() => ({ previewSegment: vi.fn() }));
vi.mock("./TranscriptPlayer", async () => {
  const { useImperativeHandle } = await import("react");
  return {
    TranscriptPlayer: (
      props: TranscriptPlayerProps & { ref?: React.Ref<TranscriptPlayerHandle> },
    ) => {
      useImperativeHandle(props.ref, () => ({
        playSegment: () => {},
        previewSegment: player.previewSegment,
        focusCursorRow: () => null,
      }));
      return (
        <div>{props.segments.map((seg, index) => (
          <button key={index} onClick={(event) => {
            event.currentTarget.focus();
            props.onSpeakerClick?.(index, new DOMRect(0, 0, 100, 20));
          }}>Speaker {index}: {seg.speaker}</button>
        ))}</div>
      );
    },
  };
});

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
// jsdom has no scrollIntoView, so give vi.spyOn a function to wrap.
Element.prototype.scrollIntoView ??= () => {};

function body() {
  return new URLSearchParams(String(vi.mocked(apiFetch).mock.calls[0][1]?.body));
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("CSS", { escape: (s: string) => s.replaceAll(":", "\\:") });
  vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
  // clearAllMocks keeps queued Once implementations; reset so none can leak.
  vi.mocked(apiFetch).mockReset();
  vi.mocked(apiFetch).mockResolvedValue({
    json: async () => ({ segments, progress: { verified: 0, total: 2 }, labels: [] }),
  } as unknown as Response);
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

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

it("offers undo for a segment relabel and adopts the undo result", async () => {
  const expiresAt = new Date(Date.now() + 300_000).toISOString();
  const relabeled = segments.map((seg, index) =>
    index === 1 ? { ...seg, speaker: "Bob" } : seg);
  vi.mocked(apiFetch)
    .mockResolvedValueOnce({
      json: async () => ({
        segments: relabeled,
        progress: { verified: 0, total: 2 },
        undo: { kind: "relabel", decisionId: "dec-9", expiresAt },
      }),
    } as unknown as Response)
    .mockResolvedValueOnce({
      json: async () => ({
        segments, progress: { verified: 0, total: 2 }, labels: defaultLabelStates,
      }),
    } as unknown as Response);
  setup({ claimCsrf: "claim-csrf" });
  fireEvent.click(screen.getByRole("radio", { name: "Just this segment" }));
  fireEvent.click(screen.getByRole("option", { name: "Bob" }));
  await screen.findByText("Segment speaker changed.");
  expect(screen.getByRole("button", { name: "Speaker 1: Bob" })).toBeTruthy();
  expect(screen.getByText("Assigned to Bob.")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Undo" }));

  await screen.findByRole("button", { name: "Speaker 1: Alice" });
  expect(vi.mocked(apiFetch).mock.calls[1][0]).toBe("/review/run/undo/relabel");
  const undoBody = new URLSearchParams(String(vi.mocked(apiFetch).mock.calls[1][1]?.body));
  expect(undoBody.get("decision_id")).toBe("dec-9");
  expect(undoBody.get("nonce")).toBe("undo:dec-9");
  expect(undoBody.get("csrf_token")).toBe("claim-csrf");
  expect(screen.queryByText("Segment speaker changed.")).toBeNull();
  // The live region no longer claims the undone assignment.
  expect(screen.queryByText("Assigned to Bob.")).toBeNull();
});

function line(index: number, label: string, speaker: string, seconds = 1, extra = {}) {
  return {
    ...segments[0], start: index * 10, end: index * 10 + seconds, label, speaker,
    segmentId: `seg-${index}`, sourceSegmentId: `seg-${index}`, ...extra,
  };
}

function labelState(label: string, resolution: string, speakerId: string | null, speakerName: string | null) {
  return { ...defaultLabelStates[0], label, resolution, speakerId, speakerName };
}

const roster = [
  { id: "alice", displayName: "Alice" },
  { id: "bob", displayName: "Bob" },
  { id: "voice2", displayName: "Voice 2" },
  { id: "cass", displayName: "Cass" },
];

function openComparison(lines: Segment[], labelStates: ReturnType<typeof labelState>[], at: number) {
  player.previewSegment.mockClear();
  render(<MediaEditor
    mediaId="media" runId="run" mediaUrl="/audio" segments={lines}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 60 }}
    lowConfidenceThreshold={0.5} reviewToken="claim"
    initialProgress={{ verified: 0, total: lines.length }}
    speakers={roster} labelStates={labelStates}
  />);
  fireEvent.click(screen.getByRole("button", { name: `Speaker ${at}: ${lines[at].speaker}` }));
}

function compareNames(): string[] {
  return screen.queryAllByRole("button", { name: /^Hear (?!this voice)/ })
    .map((button) => button.getAttribute("aria-label") ?? "");
}

it("compares against roster voices speaking under other labels", () => {
  const lines = [
    line(0, "S0", "Alice"),           // machine-matched to Alice, short
    line(1, "S1", "Bob"),             // operator-assigned to Bob
    line(2, "S0", "Alice", 3),        // machine, long
    line(3, "S2", "Voice 2"),         // unresolved placeholder, not roster Voice 2
    line(4, "S3", "Alice", 1),        // operator-assigned Alice, short
    line(5, "S3", "Alice", 4),        // operator-assigned Alice, long: preferred
  ];
  const states = [
    labelState("S0", "grounded_cosine", "alice", "Alice"),
    labelState("S1", "human_assign", "bob", "Bob"),
    labelState("S2", "unresolved", null, null),
    labelState("S3", "human_assign", "alice", "Alice"),
  ];
  openComparison(lines, states, 0);

  // Alice is the speaker on the opened line, yet she is offered: her S3 lines
  // are a different voice cluster. The placeholder "Voice 2" maps to nobody.
  expect(compareNames()).toEqual(["Hear Alice", "Hear Bob"]);

  fireEvent.click(screen.getByRole("button", { name: "Hear Alice" }));
  expect(player.previewSegment).toHaveBeenLastCalledWith(5);
  fireEvent.click(screen.getByRole("button", { name: "Hear Bob" }));
  expect(player.previewSegment).toHaveBeenLastCalledWith(1);
  expect(screen.getByRole("dialog")).toBeTruthy();
});

it("leaves out voices heard only under the opened line's label", () => {
  const lines = [line(0, "S0", "Alice"), line(1, "S0", "Bob"), line(2, "S1", "Cass")];
  const states = [
    labelState("S0", "human_assign", "alice", "Alice"),
    labelState("S1", "human_assign", "cass", "Cass"),
  ];
  openComparison(lines, states, 0);

  expect(compareNames()).toEqual(["Hear Cass"]);
});

it("maps a whole-segment override by the name it shows", () => {
  const lines = [line(0, "S0", "Alice"), line(1, "S1", "Bob"), line(2, "S1", "Alice")];
  const states = [
    labelState("S0", "human_assign", "alice", "Alice"),
    // S1 resolves to Cass, but its lines were overridden to Bob and Alice.
    labelState("S1", "human_assign", "cass", "Cass"),
  ];
  openComparison(lines, states, 0);

  expect(compareNames()).toEqual(["Hear Alice", "Hear Bob"]);
  fireEvent.click(screen.getByRole("button", { name: "Hear Alice" }));
  expect(player.previewSegment).toHaveBeenLastCalledWith(2);
});

it("uses a split child's own override", () => {
  const lines = [
    line(0, "S0", "Alice"),
    line(1, "S1", "Bob", 1, { wordStart: 0, wordEnd: 2, wordRangeSpeakerId: "bob" }),
    line(2, "S1", "Voice 2", 1, { wordStart: 2, wordEnd: 4 }),
  ];
  const states = [
    labelState("S0", "human_assign", "alice", "Alice"),
    labelState("S1", "unresolved", null, null),
  ];
  openComparison(lines, states, 0);

  expect(compareNames()).toEqual(["Hear Bob"]);
});

it("offers no comparison when playback cannot seek", () => {
  setup({
    segments: [{ ...segments[0], speaker: "Bob" }, segments[1]],
    capability: { seekEnabled: false, reasons: [], mediaDuration: 2 },
    speakers: [{ id: "alice", displayName: "Alice" }, { id: "bob", displayName: "Bob" }],
  });
  expect(screen.queryByRole("button", { name: "Hear Bob" })).toBeNull();
});

function renderUnclaimed(overrides: Partial<MediaEditorProps> = {}) {
  render(<MediaEditor
    mediaId="media" runId="run" mediaUrl="/audio"
    segments={segments}
    capability={{ seekEnabled: true, reasons: [], mediaDuration: 2 }}
    lowConfidenceThreshold={0.5}
    reviewToken={null}
    initialProgress={{ verified: 0, total: 2 }}
    speakers={[]}
    claimCsrf="claim-csrf"
    {...overrides}
  />);
}

it("auto-claims the editor for a single operator and adopts the token", async () => {
  vi.mocked(apiFetch).mockResolvedValue({
    json: async () => ({ token: "fresh-token", tagCsrf: "tag", clipCsrf: "clip" }),
  } as unknown as Response);
  renderUnclaimed();
  await waitFor(() => expect(apiFetch).toHaveBeenCalledOnce());
  const [url, init] = vi.mocked(apiFetch).mock.calls[0];
  expect(url).toBe("/media/media/editor/claim");
  expect(init?.method).toBe("POST");
  expect(init?.headers).toMatchObject({
    "content-type": "application/x-www-form-urlencoded",
  });
  expect(body().get("run_id")).toBe("run");
  expect(body().get("csrf_token")).toBe("claim-csrf");
  await waitFor(() =>
    expect(new URLSearchParams(window.location.search).get("token")).toBe("fresh-token"));
  expect(screen.queryByText("Read-only view.", { exact: false })).toBeNull();
});

it.each([
  ["several operators share the console", { multiUser: true }],
  ["a review token is already held", { reviewToken: "held" }],
])("does not auto-claim when %s", async (_case, overrides) => {
  renderUnclaimed(overrides);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(apiFetch).not.toHaveBeenCalled();
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
