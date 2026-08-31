import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import {
  QuoteBoardIsland,
  type QuoteBoardProps,
} from "../components/QuoteBoardIsland";
import { readProps } from "../lib/mount";

export function mount(el: HTMLElement): void {
  const props = readProps<QuoteBoardProps>(el);
  createRoot(el).render(
    <StrictMode>
      <QuoteBoardIsland {...props} />
    </StrictMode>,
  );
}
