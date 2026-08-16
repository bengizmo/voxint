import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { TranscriptPlayer, type TranscriptPlayerProps } from "../components/TranscriptPlayer";
import { readProps } from "../lib/mount";

// Island entry: reads server-rendered props off the mount node and renders the
// React component OVER the server fallback markup already inside the div. The
// shared loader (main.ts) calls this `mount` once per matching node.
export function mount(el: HTMLElement): void {
  const props = readProps<TranscriptPlayerProps>(el);
  createRoot(el).render(
    <StrictMode>
      <TranscriptPlayer {...props} />
    </StrictMode>,
  );
}
