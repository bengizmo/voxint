import { useCallback, useRef } from "react";

import type { PlaybackCapability } from "../lib/playback";
import type { Turn } from "../lib/peaks";
import type { OutlineProps } from "../lib/outline";
import { OutlinePanel } from "./OutlinePanel";
import {
  type Segment,
  TranscriptPlayer,
  type TranscriptPlayerHandle,
} from "./TranscriptPlayer";

export interface MediaEditorProps {
  runId: string;
  mediaUrl: string;
  segments: Segment[];
  capability: PlaybackCapability;
  lowConfidenceThreshold: number;
  reviewToken: string | null;
  initialProgress: { verified: number; total: number };
  peaksUrl?: string | null;
  turns?: Turn[];
  speakers: { id: string; displayName: string }[];
  outline?: OutlineProps;
}

export function MediaEditor(props: MediaEditorProps): React.JSX.Element {
  const {
    runId,
    mediaUrl,
    segments,
    capability,
    lowConfidenceThreshold,
    initialProgress,
    peaksUrl,
    turns,
    outline,
  } = props;

  const playerRef = useRef<TranscriptPlayerHandle>(null);

  const verified = initialProgress.verified;
  const total = initialProgress.total;
  const pct = total > 0 ? Math.round((verified / total) * 100) : 0;

  const handleOutlineJump = useCallback(
    (index: number) => playerRef.current?.playSegment(index),
    [],
  );

  return (
    <>
      <OutlinePanel
        outline={outline}
        segments={segments}
        capability={capability}
        onJump={handleOutlineJump}
      />

      <div className="me-toolbar">
        <span className="me-progress" aria-live="polite">
          {verified}/{total} verified ({pct}%)
        </span>
      </div>

      <div className="lib-two-col">
        <div className="lib-main">
          <TranscriptPlayer
            ref={playerRef}
            runId={runId}
            mediaUrl={mediaUrl}
            segments={segments}
            capability={capability}
            lowConfidenceThreshold={lowConfidenceThreshold}
            peaksUrl={peaksUrl}
            turns={turns}
          />
        </div>

        <div className="lib-rail">
          <div className="card">
            <h3>Speakers</h3>
            <p className="text-sm text-muted">
              Speaker rail ships in a later slice.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
