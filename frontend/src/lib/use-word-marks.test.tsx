// @vitest-environment jsdom
import { act, cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { apiFetch, ApiError } from "./api-client";
import { useWordMarks } from "./use-word-marks";
import type { WordMarksPayload } from "./word-marks";

vi.mock("./api-client", async (original) => ({ ...await original<typeof import("./api-client")>(), apiFetch: vi.fn() }));
afterEach(() => { cleanup(); vi.mocked(apiFetch).mockReset(); });
const payload: WordMarksPayload = { runId: "run", version: "1.hash", detectionError: null, fillerListDefault: true, emissions: [], stale: [] };
const response = (data: unknown) => ({ json: async () => data }) as Response;
function deferred() {
  let resolve!: (value: Response) => void;
  const promise = new Promise<Response>((done) => { resolve = done; });
  return { promise, resolve };
}

it("drops late snapshots and errors, and preserves snapshot identity for equal versions at the same focus", async () => {
  const first = deferred();
  vi.mocked(apiFetch).mockReturnValueOnce(first.promise).mockResolvedValue(response(payload));
  const error = vi.fn();
  let marks!: ReturnType<typeof useWordMarks>;
  let renders = 0;
  function Probe() { renders += 1; marks = useWordMarks("run", error); return null; }
  render(<Probe />);
  await act(async () => { await marks.refresh("seg"); });
  expect(marks.snapshot?.focus).toBe("seg");
  const snapshot = marks.snapshot;
  const before = renders;
  await act(async () => { await marks.refresh("seg"); });
  expect(marks.snapshot).toBe(snapshot);
  expect(renders).toBe(before);
  await act(async () => { first.resolve(response({ ...payload, version: "0.old" })); });
  expect(marks.snapshot).toBe(snapshot);
  expect(renders).toBe(before);
  await act(async () => { await marks.refresh(null); });
  expect(marks.snapshot?.focus).toBeNull();
  expect(marks.snapshot).not.toBe(snapshot);

  const staleError = deferred();
  vi.mocked(apiFetch).mockReturnValueOnce(staleError.promise);
  let failed!: Promise<void>;
  act(() => { failed = marks.refresh("old"); });
  await act(async () => { await marks.refresh("new"); });
  await act(async () => {
    staleError.resolve({ json: async () => { throw new ApiError(500, "outdated failure"); } } as unknown as Response);
    await failed;
  });
  expect(error).not.toHaveBeenCalled();
  expect(marks.snapshot?.focus).toBe("new");
});

it("invalidates pending reads and adopts writes before flushing a deferred focus read", async () => {
  vi.mocked(apiFetch).mockResolvedValue(response(payload));
  const error = vi.fn();
  let marks!: ReturnType<typeof useWordMarks>;
  function Probe() { marks = useWordMarks("run", error); return null; }
  render(<Probe />);
  await waitFor(() => expect(marks.snapshot).not.toBeNull());
  const oldRead = deferred();
  vi.mocked(apiFetch).mockReturnValueOnce(oldRead.promise);
  let reading!: Promise<void>;
  act(() => { reading = marks.refresh(null); });
  const ticket = marks.beginWrite();
  const written = { ...payload, version: "2.hash" };
  act(() => { marks.adopt(written, null, ticket); marks.finishWrite(true); });
  expect(marks.snapshot?.payload).toBe(written);
  await act(async () => { oldRead.resolve(response(payload)); await reading; });
  expect(marks.snapshot?.payload).toBe(written);

  const write = marks.beginWrite();
  const fresh = { ...payload, version: "3.hash" };
  vi.mocked(apiFetch).mockResolvedValue(response(fresh));
  const before = vi.mocked(apiFetch).mock.calls.length;
  await act(async () => { await marks.refresh("new"); });
  expect(vi.mocked(apiFetch).mock.calls.length).toBe(before);
  act(() => { marks.adopt(written, "old", write); });
  expect(marks.snapshot?.payload).toBe(written);
  expect(marks.snapshot?.focus).toBe("old");
  await act(async () => { marks.finishWrite(true); });
  expect(marks.snapshot?.payload.version).toBe("3.hash");
  expect(marks.snapshot?.focus).toBe("new");
  expect(vi.mocked(apiFetch).mock.calls.length).toBe(before + 1);
});

it("coalesces deferred focus changes and refreshes without relying on a busy-state render", async () => {
  vi.mocked(apiFetch).mockResolvedValue(response(payload));
  const error = vi.fn();
  let marks!: ReturnType<typeof useWordMarks>;
  function Probe() { marks = useWordMarks("run", error); return null; }
  render(<Probe />);
  await waitFor(() => expect(marks.snapshot).not.toBeNull());
  await act(async () => { await marks.refresh("original"); });
  const before = vi.mocked(apiFetch).mock.calls.length;
  const write = marks.beginWrite();
  act(() => {
    marks.requestFocus("intermediate");
    marks.requestFocus("latest");
    void marks.refresh();
  });
  expect(vi.mocked(apiFetch).mock.calls.length).toBe(before);
  await act(async () => {
    marks.adopt({ ...payload, version: "2.hash" }, "original", write);
    marks.finishWrite(true);
  });
  expect(vi.mocked(apiFetch).mock.calls.slice(before).map(([url]) => url))
    .toEqual(["/review/run/word-marks?segment=latest"]);

  const returnWrite = marks.beginWrite();
  act(() => {
    marks.requestFocus("intermediate");
    marks.requestFocus("latest");
  });
  const reads = vi.mocked(apiFetch).mock.calls.length;
  act(() => {
    marks.adopt({ ...payload, version: "3.hash" }, "latest", returnWrite);
    marks.finishWrite(true);
  });
  expect(vi.mocked(apiFetch).mock.calls.length).toBe(reads);
  expect(marks.snapshot?.payload.version).toBe("3.hash");
});

it.each([false, true])("refreshes desired focus once after a failed write (pending read=%s)", async (pendingRead) => {
  vi.mocked(apiFetch).mockResolvedValue(response(payload));
  let marks!: ReturnType<typeof useWordMarks>;
  const error = vi.fn();
  function Probe() { marks = useWordMarks("run", error); return null; }
  render(<Probe />);
  await act(async () => { await marks.refresh("seg"); });
  const oldRead = deferred();
  let reading: Promise<void> | undefined;
  if (pendingRead) {
    vi.mocked(apiFetch).mockReturnValueOnce(oldRead.promise);
    act(() => { reading = marks.refresh("seg"); });
  }
  marks.beginWrite();
  const before = vi.mocked(apiFetch).mock.calls.length;
  const fresh = { ...payload, version: "2.hash" };
  vi.mocked(apiFetch).mockResolvedValue(response(fresh));
  await act(async () => { marks.finishWrite(false); });
  expect(vi.mocked(apiFetch).mock.calls.slice(before).map(([url]) => url))
    .toEqual(["/review/run/word-marks?segment=seg"]);
  expect(marks.snapshot?.payload).toBe(fresh);
  if (pendingRead) await act(async () => { oldRead.resolve(response(payload)); await reading; });
  expect(marks.snapshot?.payload).toBe(fresh);
});

it("makes retained callbacks terminal on unmount, including pending read/write settlement", async () => {
  const read = deferred();
  vi.mocked(apiFetch).mockReturnValue(read.promise);
  const error = vi.fn();
  let marks!: ReturnType<typeof useWordMarks>;
  function Probe() { marks = useWordMarks("run", error); return null; }
  const view = render(<Probe />);
  const retained = marks;
  const write = retained.beginWrite();
  retained.requestFocus("seg");
  view.unmount();
  await act(async () => {
    retained.adopt(payload, "seg", write);
    retained.finishWrite(false);
    await retained.refresh("seg");
    read.resolve(response(payload));
  });
  expect(vi.mocked(apiFetch)).toHaveBeenCalledOnce();
  expect(error).not.toHaveBeenCalled();
});

it("accepts identical version/focus on a new run and ignores old-run write callbacks", async () => {
  vi.mocked(apiFetch).mockResolvedValue(response(payload));
  const error = vi.fn();
  let marks!: ReturnType<typeof useWordMarks>;
  function Probe({ runId }: { runId: string }) { marks = useWordMarks(runId, error); return null; }
  const view = render(<Probe runId="run" />);
  await waitFor(() => expect(marks.snapshot?.payload.runId).toBe("run"));
  const retained = marks;
  const write = retained.beginWrite();
  retained.requestFocus("old-focus");
  vi.mocked(apiFetch).mockResolvedValue(response({ ...payload, runId: "next" }));
  view.rerender(<Probe runId="next" />);
  await waitFor(() => expect(marks.snapshot?.payload.runId).toBe("next"));
  const before = vi.mocked(apiFetch).mock.calls.length;
  await act(async () => {
    retained.adopt(payload, "old-focus", write);
    retained.finishWrite(false);
    await retained.refresh();
  });
  expect(vi.mocked(apiFetch).mock.calls.length).toBe(before);
  expect(marks.snapshot?.payload.runId).toBe("next");
});
