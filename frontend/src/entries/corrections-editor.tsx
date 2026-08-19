import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { CorrectionsEditor, type CorrectionsEditorProps } from "../components/CorrectionsEditor";
import { readProps } from "../lib/mount";

// Island entry (issue #84): the console corrections editor. Mounted on the
// Settings page over its server-rendered read-only fallback. The shared loader
// (main.ts) calls `mount` once.
export function mount(el: HTMLElement): void {
  const props = readProps<CorrectionsEditorProps>(el);
  createRoot(el).render(
    <StrictMode>
      <CorrectionsEditor {...props} />
    </StrictMode>,
  );
}
