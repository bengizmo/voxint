import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { WorkbenchPlayer, type WorkbenchPlayerProps } from "../components/WorkbenchPlayer";
import { readProps } from "../lib/mount";

// Island entry (issues #49 + #55): reads server-rendered props off the mount
// node and renders the React player OVER the bare <audio> fallback already
// inside the div. The shared loader (main.ts) calls this `mount` once per node.
export function mount(el: HTMLElement): void {
  const props = readProps<WorkbenchPlayerProps>(el);
  createRoot(el).render(
    <StrictMode>
      <WorkbenchPlayer {...props} />
    </StrictMode>,
  );
}
