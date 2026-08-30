import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  ExploreIsland,
  type ExploreIslandProps,
} from "../components/ExploreIsland";
import { readProps } from "../lib/mount";

export function mount(el: HTMLElement): void {
  const props = readProps<ExploreIslandProps>(el);
  createRoot(el).render(
    <StrictMode>
      <ExploreIsland {...props} />
    </StrictMode>,
  );
}
