// A roster speaker's voice from ANOTHER recording (issue #714), played in the
// speaker menu so the operator can compare before assigning.
//
// The clip is fetched whole and played from a blob on its own element, never
// by pointing an <audio src> at the route: only a fetch can tell "no confirmed
// line" (404) from "the audio is gone" (410), and the editor says which.
//
// The contract: the latest request wins. A newer play(), a stop() or a
// dispose() supersedes whatever is in flight, and a superseded call has no
// side effects (no object URL, no playback, no onStart).

import { getStoredRate } from "./playback";

export type VoiceSampleOutcome =
  | "playing"
  // A newer request, stop() or dispose() overtook this one; say nothing.
  | "superseded"
  // 404 no_voice_sample: no confirmed line elsewhere.
  | "none"
  // 410: confirmed lines exist, but no recording holding one can be played.
  | "gone"
  | "failed";

export interface VoiceSamplePlayer {
  play(url: string): Promise<VoiceSampleOutcome>;
  // The main player took over: drop any in-flight request, pause the sample.
  stop(): void;
  // The editor is unmounting: stop, and release the blob.
  dispose(): void;
}

// `onStart` runs just before a sample starts, so the caller can pause the main
// player: lib/playback.ts does not make playback exclusive on its own.
export function createVoiceSamplePlayer(
  onStart: () => void,
): VoiceSamplePlayer {
  let request: AbortController | null = null;
  let audio: HTMLAudioElement | null = null;
  let objectUrl: string | null = null;
  let disposed = false;

  const stop = (): void => {
    request?.abort();
    request = null;
    audio?.pause();
  };

  return {
    async play(url) {
      if (disposed) return "superseded";
      request?.abort();
      const controller = new AbortController();
      request = controller;
      // Re-checked after every await: an answer that arrives late must not
      // replace the clip a newer click asked for.
      const stale = (): boolean =>
        controller.signal.aborted || request !== controller;
      try {
        const response = await fetch(url, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (stale()) return "superseded";
        if (response.status === 410) return "gone";
        if (response.status === 404) {
          // The route also answers 404 "not found" when the current recording
          // no longer exists; only its own code means "no confirmed line".
          const body = (await response.json()) as { detail?: unknown } | null;
          if (stale()) return "superseded";
          return body?.detail === "no_voice_sample" ? "none" : "failed";
        }
        if (response.status !== 200) return "failed";
        const blob = await response.blob();
        if (stale()) return "superseded";
        audio ??= new Audio();
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(blob);
        audio.src = objectUrl;
        // After `src`: loading a new source resets the rate to the default.
        // The stored rate is the main player's, so both voices are heard at
        // the same speed.
        audio.playbackRate = getStoredRate();
        onStart();
        await audio.play();
        return stale() ? "superseded" : "playing";
      } catch (error) {
        // An aborted fetch, or a play() interrupted by a pause or a newer
        // source, is not a failure the operator needs to hear about. jsdom's
        // DOMException is not an Error, so match on the name.
        const aborted =
          error !== null &&
          typeof error === "object" &&
          "name" in error &&
          error.name === "AbortError";
        return stale() || aborted ? "superseded" : "failed";
      }
    },
    stop,
    dispose() {
      disposed = true;
      stop();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      audio?.removeAttribute("src");
    },
  };
}
