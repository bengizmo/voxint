import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { ReviewStepper, type ReviewStepperProps } from "../components/ReviewStepper";
import { readProps } from "../lib/mount";

// Island entry (issue #53): the verify-and-advance review loop. Mounted only on
// the claim-gated /review/{id}/transcript surface, over its server-rendered
// flagged-segment fallback. The shared loader (main.ts) calls `mount` once.
export function mount(el: HTMLElement): void {
  const props = readProps<ReviewStepperProps>(el);
  createRoot(el).render(
    <StrictMode>
      <ReviewStepper {...props} />
    </StrictMode>,
  );
}
