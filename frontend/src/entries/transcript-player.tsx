import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { TranscriptPlayer, type TranscriptPlayerProps } from "../components/TranscriptPlayer";
import { parseJumpParam } from "../lib/jump";
import { readProps } from "../lib/mount";

// Island entry: reads server-rendered props off the mount node and renders the
// React component OVER the server fallback markup already inside the div. The
// shared loader (main.ts) calls this `mount` once per matching node.
export function mount(el: HTMLElement): void {
  const props = readProps<TranscriptPlayerProps>(el);
  // Deep-link jump (issue #121): a Meaning search result opens the transcript at
  // ?t=SECONDS. Resolve it here — only the read-only transcript surface uses this
  // entry, so the review workbench (a different entry) never jumps. A missing or
  // invalid ?t= is null → the component treats it as no jump.
  const jumpToSeconds = parseJumpParam(window.location.search);
  createRoot(el).render(
    <StrictMode>
      <TranscriptPlayer {...props} jumpToSeconds={jumpToSeconds} />
    </StrictMode>,
  );
}
