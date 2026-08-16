import { useEffect, useRef, useState } from "react";

import {
  cancelActiveTurn,
  getStoredRate,
  playTurn,
  setStoredRate,
  type PlaybackCapability,
} from "../lib/playback";
import { CapabilityBanner, SpeedControl } from "./PlaybackControls";

export interface WorkbenchPlayerProps {
  runId: string;
  mediaUrl: string;
  capability: PlaybackCapability;
}

// Workbench audio bridge (issues #49 + #55). Mounted OUTSIDE #labels, it owns the
// <audio> element, the speed control, and the visible capability banner. The
// per-turn seek buttons live INSIDE #labels — server-rendered and replaced on
// every htmx innerHTML swap — so this island drives them with a DOCUMENT-level
// delegated click listener (survives swaps) plus an htmx:afterSwap re-enable pass
// (the server always re-renders them disabled, an honest JS-off default).
export function WorkbenchPlayer({ runId, mediaUrl, capability }: WorkbenchPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [rate, setRate] = useState<number>(() => getStoredRate());

  // Keep the element's rate in sync with the (persisted) control.
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) audio.playbackRate = rate;
  }, [rate]);

  useEffect(() => {
    const seekEnabled = capability.seekEnabled;
    const disabledReason = capability.reasons[0]?.message ?? "Playback is unavailable.";

    // Enable pass: reconcile every server-rendered seek button in #labels with
    // the capability. Run on mount and after each relevant swap. NOTE: this uses
    // the capability captured at page load. If the media were reclaimed mid-
    // session (rare on a single-operator local box), a post-swap enable pass
    // would re-enable buttons against stale state — clicking one then no-ops (the
    // <audio> 410s and play() rejects), so the failure is a dead button, never a
    // wrong-voice seek. Re-fetching capability per swap isn't worth the ceremony.
    const enablePass = (): void => {
      const labels = document.getElementById("labels");
      if (!labels) return;
      const buttons = labels.querySelectorAll<HTMLButtonElement>("[data-voxint-seek]");
      buttons.forEach((btn) => {
        if (seekEnabled) {
          btn.disabled = false;
          btn.removeAttribute("title");
        } else {
          btn.disabled = true;
          btn.title = disabledReason;
        }
      });
    };

    // Delegated click: a seek button anywhere inside the CURRENT #labels.
    const onClick = (event: MouseEvent): void => {
      const labels = document.getElementById("labels");
      if (!labels) return;
      const origin = event.target;
      if (!(origin instanceof Element)) return;
      const btn = origin.closest<HTMLButtonElement>("[data-voxint-seek]");
      if (!btn || !labels.contains(btn) || btn.disabled) return;
      const start = Number.parseFloat(btn.dataset.seekStart ?? "");
      const end = Number.parseFloat(btn.dataset.seekEnd ?? "");
      // Re-validate at click time: the data attributes are server-rendered text.
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
      const audio = audioRef.current;
      if (audio) playTurn(audio, start, end);
    };

    // htmx re-renders #labels innerHTML after every decision; re-run the enable
    // pass only for swaps that touch #labels.
    const onAfterSwap = (event: Event): void => {
      const labels = document.getElementById("labels");
      if (!labels) return;
      const target = (event as CustomEvent<{ target?: EventTarget }>).detail?.target;
      if (!(target instanceof Element)) return;
      if (target === labels || labels.contains(target) || target.contains(labels)) {
        enablePass();
      }
    };

    enablePass();
    document.addEventListener("click", onClick);
    document.addEventListener("htmx:afterSwap", onAfterSwap);
    return () => {
      // Symmetric teardown — StrictMode double-invokes this in dev; never leak a
      // listener or a turn guard.
      document.removeEventListener("click", onClick);
      document.removeEventListener("htmx:afterSwap", onAfterSwap);
      cancelActiveTurn();
    };
  }, [capability]);

  const onRateChange = (next: number): void => {
    setRate(next);
    setStoredRate(next);
  };

  return (
    <div>
      <div className="flex items-center my-2">
        <SpeedControl rate={rate} onChange={onRateChange} />
      </div>
      <audio
        ref={audioRef}
        controls
        preload="metadata"
        src={mediaUrl}
        className="w-full"
        data-run-id={runId}
      >
        Your browser does not support the audio element.
      </audio>
      <CapabilityBanner capability={capability} />
    </div>
  );
}
