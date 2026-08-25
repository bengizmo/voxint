import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  ReadOnlyTranscript,
  type ReadOnlyTranscriptProps,
} from "../components/ReadOnlyTranscript";
import { parseJumpParam } from "../lib/jump";
import { readProps } from "../lib/mount";

// Island entry: reads server-rendered props off the mount node and renders the
// React tree OVER the server fallback markup already inside the div (createRoot
// REPLACES it on first render — this is not hydrateRoot). The read-only surface
// composes the OutlinePanel above the player (issue #87), so this entry mounts
// ReadOnlyTranscript rather than the bare player. The shared loader (main.ts)
// calls this `mount` once per matching node.
export function mount(el: HTMLElement): void {
  const props = readProps<ReadOnlyTranscriptProps>(el);
  // Deep-link jump (issue #121): a Meaning search result opens the transcript at
  // ?t=SECONDS. Resolve it here — only the read-only transcript surface uses this
  // entry, so the review workbench (a different entry) never jumps. A missing or
  // invalid ?t= is null → the component treats it as no jump.
  const jumpToSeconds = parseJumpParam(window.location.search);
  createRoot(el).render(
    <StrictMode>
      <ReadOnlyTranscript {...props} jumpToSeconds={jumpToSeconds} />
    </StrictMode>,
  );
}
