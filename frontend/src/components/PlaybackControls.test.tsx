// @vitest-environment jsdom
import { useRef } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AudioTransport } from "./PlaybackControls";
import { playTurn } from "../lib/playback";

function Player({ src = "/recording.wav" }: { src?: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  return (
    <>
      <audio ref={audioRef} src={src} hidden />
      <AudioTransport audioRef={audioRef} mediaUrl={src} />
    </>
  );
}
function setup() {
  const result = render(<Player />);
  const audio = result.container.querySelector("audio")!;
  vi.spyOn(audio, "play").mockImplementation(async () => {
    Object.defineProperty(audio, "paused", {
      configurable: true,
      value: false,
    });
    fireEvent.play(audio);
  });
  vi.spyOn(audio, "pause").mockImplementation(() => {
    Object.defineProperty(audio, "paused", { configurable: true, value: true });
    fireEvent.pause(audio);
  });
  return { ...result, audio };
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
describe("audio transport", () => {
  it("tracks external playback, pauses, duration and ending", async () => {
    const { audio } = setup();
    expect((screen.getByRole("slider") as HTMLInputElement).disabled).toBe(
      true,
    );
    Object.defineProperty(audio, "duration", {
      configurable: true,
      value: 120,
    });
    fireEvent.loadedMetadata(audio);
    expect((screen.getByRole("slider") as HTMLInputElement).disabled).toBe(
      false,
    );
    await act(async () => {
      await audio.play();
    });
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    expect(audio.pause).toHaveBeenCalled();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Play" }));
    });
    audio.currentTime = 65;
    fireEvent.timeUpdate(audio);
    expect(screen.getByRole("slider").getAttribute("aria-valuetext")).toBe(
      "1:05 of 2:00",
    );
    Object.defineProperty(audio, "ended", { configurable: true, value: true });
    fireEvent.ended(audio);
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
  });
  it("manual seeking cancels the segment end guard", async () => {
    const { audio } = setup();
    Object.defineProperty(audio, "duration", {
      configurable: true,
      value: 120,
    });
    fireEvent.loadedMetadata(audio);
    await act(async () => {
      playTurn(audio, 0, 5);
    });
    fireEvent.change(
      screen.getByRole("slider", { name: "Playback position" }),
      { target: { value: "30" } },
    );
    fireEvent.timeUpdate(audio);
    expect(audio.currentTime).toBe(30);
    expect(audio.pause).not.toHaveBeenCalled();
  });
  it("reports failures and resets on a new source or empty media", async () => {
    const { audio, rerender } = setup();
    vi.mocked(audio.play).mockRejectedValue(new Error("blocked"));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Play" }));
    });
    expect(screen.getByRole("status").textContent).toContain(
      "Playback could not start",
    );
    fireEvent.error(audio);
    expect(screen.getByRole("status").textContent).toContain(
      "could not be loaded",
    );
    rerender(<Player src="/other.wav" />);
    expect(screen.queryByRole("status")).toBeNull();
    Object.defineProperty(audio, "duration", {
      configurable: true,
      value: Infinity,
    });
    fireEvent.durationChange(audio);
    expect((screen.getByRole("slider") as HTMLInputElement).disabled).toBe(
      true,
    );
    fireEvent.emptied(audio);
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
  });
});
