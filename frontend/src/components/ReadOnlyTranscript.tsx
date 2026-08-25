import { useCallback, useRef } from "react";

import type { OutlineProps } from "../lib/outline";
import { OutlinePanel } from "./OutlinePanel";
import {
  TranscriptPlayer,
  type TranscriptPlayerHandle,
  type TranscriptPlayerProps,
} from "./TranscriptPlayer";

export interface ReadOnlyTranscriptProps extends TranscriptPlayerProps {
  // Navigable outline (issue #87): grounded entity-mention jump targets plus
  // inert summary/topics context, from the shared transcript island props. The
  // read-only run transcript renders the SAME OutlinePanel the review workbench
  // does — minus every write control — wired to the player. Defaulted, so a
  // props payload without it simply renders no panel.
  outline?: OutlineProps;
}

// Read-only transcript surface (issue #87 follow-up): the OutlinePanel above the
// pure TranscriptPlayer, wired so an outline jump reveals — and, when seek is
// trusted, plays — the resolved line, the same affordance the review stepper
// gives without any of its writes. The panel lives HERE, not inside the player,
// so TranscriptPlayer stays a pure leaf and the review surface (which renders its
// own OutlinePanel) never doubles it up. Segments are static on this surface (no
// splits/edits without a claim), so the panel resolves against props.segments.
export function ReadOnlyTranscript({
  outline,
  ...playerProps
}: ReadOnlyTranscriptProps): React.JSX.Element {
  const playerRef = useRef<TranscriptPlayerHandle>(null);
  // OutlinePanel resolves a mention's time to a current line and hands us the
  // index; playSegment scrolls it into view and plays from there when seek is
  // available (the jump stays a reading act when seek is not — see OutlinePanel).
  const onJump = useCallback((index: number) => {
    playerRef.current?.playSegment(index);
  }, []);
  return (
    <>
      <OutlinePanel
        outline={outline}
        segments={playerProps.segments}
        capability={playerProps.capability}
        onJump={onJump}
      />
      <TranscriptPlayer ref={playerRef} {...playerProps} />
    </>
  );
}
