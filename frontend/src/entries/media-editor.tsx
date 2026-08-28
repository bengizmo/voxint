import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { MediaEditor, type MediaEditorProps } from "../components/MediaEditor";
import { readProps } from "../lib/mount";

export function mount(el: HTMLElement): void {
  const props = readProps<MediaEditorProps>(el);
  createRoot(el).render(
    <StrictMode>
      <MediaEditor {...props} />
    </StrictMode>,
  );
}
