// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createVoiceSamplePlayer } from "./voice-sample";
import { setStoredRate } from "./playback";

const fetchMock = vi.fn<typeof fetch>();
const play = vi.fn<() => Promise<void>>();
const pause = vi.fn();
let audio: HTMLAudioElement;
const createUrl = vi.fn(() => `blob:${Math.random()}`);
const revokeUrl = vi.fn();
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
beforeEach(() => {
  audio = document.createElement("audio");
  audio.play = play.mockResolvedValue();
  audio.pause = pause;
  vi.stubGlobal(
    "Audio",
    class {
      constructor() {
        return audio;
      }
    },
  );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("URL", {
    createObjectURL: createUrl,
    revokeObjectURL: revokeUrl,
  });
  fetchMock.mockImplementation(async () => new Response(new Blob(["wav"])));
});
afterEach(() => {
  vi.resetAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
});
it("plays 200 at the stored rate and calls onStart before play", async () => {
  setStoredRate(1.5);
  const start = vi.fn();
  const sampler = createVoiceSamplePlayer(start);
  expect(await sampler.play("/sample")).toBe("playing");
  expect(audio.playbackRate).toBe(1.5);
  expect(start.mock.invocationCallOrder[0]).toBeLessThan(
    play.mock.invocationCallOrder[0],
  );
  expect(fetchMock).toHaveBeenCalledWith("/sample", {
    cache: "no-store",
    signal: expect.any(AbortSignal),
  });
});
it("aborts a rapid earlier call and never creates a URL for its late blob", async () => {
  const blob = deferred<Blob>();
  fetchMock.mockResolvedValueOnce({
    status: 200,
    blob: () => blob.promise,
  } as Response);
  const start = vi.fn();
  const sampler = createVoiceSamplePlayer(start);
  const first = sampler.play("/first");
  await Promise.resolve();
  expect(await sampler.play("/second")).toBe("playing");
  expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
  blob.resolve(new Blob(["old"]));
  expect(await first).toBe("superseded");
  expect(createUrl).toHaveBeenCalledOnce();
  expect(start).toHaveBeenCalledOnce();
  expect(play).toHaveBeenCalledOnce();
});
it("reuses one audio element and revokes URLs on replacement and dispose", async () => {
  createUrl
    .mockReturnValueOnce("blob:first")
    .mockReturnValueOnce("blob:second");
  const sampler = createVoiceSamplePlayer(vi.fn());
  await sampler.play("/first");
  await sampler.play("/second");
  expect(revokeUrl).toHaveBeenCalledWith("blob:first");
  expect(audio.src).toBe("blob:second");
  sampler.dispose();
  expect(revokeUrl).toHaveBeenLastCalledWith("blob:second");
  expect(audio.hasAttribute("src")).toBe(false);
  expect(pause).toHaveBeenCalledOnce();
});
it.each([
  [404, JSON.stringify({ detail: "no_voice_sample" }), "none"],
  [404, JSON.stringify({ detail: "not found" }), "failed"],
  [404, "not json", "failed"],
  [410, "", "gone"],
  [401, "", "failed"],
  [500, "", "failed"],
])("maps HTTP %s body %s to %s", async (status, body, expected) => {
  fetchMock.mockResolvedValue(new Response(body, { status }));
  expect(await createVoiceSamplePlayer(vi.fn()).play("/sample")).toBe(expected);
  expect(play).not.toHaveBeenCalled();
});
it("maps a network error to failed", async () => {
  fetchMock.mockRejectedValue(new TypeError("offline"));
  expect(await createVoiceSamplePlayer(vi.fn()).play("/sample")).toBe("failed");
});
it.each([
  ["NotAllowedError", "failed"],
  ["AbortError", "superseded"],
])("maps play rejection %s to %s", async (name, expected) => {
  play.mockRejectedValue(new DOMException("blocked", name));
  expect(await createVoiceSamplePlayer(vi.fn()).play("/sample")).toBe(expected);
});
it.each(["stop", "dispose"] as const)(
  "%s during fetch supersedes without playback",
  async (method) => {
    const pending = deferred<Response>();
    fetchMock.mockReturnValue(pending.promise);
    const start = vi.fn();
    const sampler = createVoiceSamplePlayer(start);
    const result = sampler.play("/sample");
    sampler[method]();
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    pending.resolve(new Response("wav"));
    expect(await result).toBe("superseded");
    expect(createUrl).not.toHaveBeenCalled();
    expect(start).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
  },
);
it("stop pauses a playing sample", async () => {
  const sampler = createVoiceSamplePlayer(vi.fn());
  await sampler.play("/sample");
  sampler.stop();
  expect(pause).toHaveBeenCalledOnce();
});
it("stop while play is pending supersedes its eventual success", async () => {
  const pending = deferred<void>();
  play.mockReturnValue(pending.promise);
  const sampler = createVoiceSamplePlayer(vi.fn());
  const result = sampler.play("/sample");
  await vi.waitFor(() => expect(play).toHaveBeenCalledOnce());
  sampler.stop();
  pending.resolve();
  expect(await result).toBe("superseded");
});
