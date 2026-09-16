import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  CommandPalette,
  type CommandPaletteProps,
} from "../components/CommandPalette";
import { readProps } from "../lib/mount";

export function mount(
  el: HTMLElement,
  opts?: { open?: boolean },
): void {
  const props = readProps<CommandPaletteProps>(el);
  createRoot(el).render(
    <StrictMode>
      <CommandPalette {...props} initialOpen={opts?.open ?? false} />
    </StrictMode>,
  );
}
