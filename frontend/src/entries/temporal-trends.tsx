import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  TemporalTrendsIsland,
  type TemporalTrendsProps,
} from "../components/TemporalTrendsIsland";
import { readProps } from "../lib/mount";

export function mount(el: HTMLElement): void {
  const props = readProps<TemporalTrendsProps>(el);
  createRoot(el).render(
    <StrictMode>
      <TemporalTrendsIsland {...props} />
    </StrictMode>,
  );
}
