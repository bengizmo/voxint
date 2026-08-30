import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  SpeakerTimelineIsland,
  type SpeakerTimeline,
} from "../components/SpeakerTimelineIsland";
import { readProps } from "../lib/mount";

export function mount(el: HTMLElement): void {
  const timeline = readProps<SpeakerTimeline>(el);
  const runId = el.dataset.runId;
  if (!runId) throw new Error("speaker timeline mount point missing run id");
  createRoot(el).render(
    <StrictMode>
      <SpeakerTimelineIsland runId={runId} timeline={timeline} />
    </StrictMode>,
  );
}
