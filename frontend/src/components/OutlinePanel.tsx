import { useCallback } from "react";

import { resolveJumpIndex } from "../lib/jump";
import { formatClock, totalDropped, type OutlineProps } from "../lib/outline";
import type { PlaybackCapability } from "../lib/playback";
import type { Segment } from "./TranscriptPlayer";

// Navigable outline (issue #87): grounded entity mentions become jump targets;
// summary and topics render as inert context. A click resolves the target's
// startSeconds to the current rendered line (resolveJumpIndex) and calls onJump,
// which scrolls the transcript and, when the player can seek, plays from there.
// The jump stays a reading act even when audio seek is unavailable, so entries
// are never disabled; only the affordance wording reflects capability.

interface OutlinePanelProps {
  outline?: OutlineProps;
  // The live rendered segments (the authority on current lines; grows on split).
  segments: Segment[];
  capability: PlaybackCapability;
  onJump: (index: number) => void;
}

const KIND_LABEL: Record<string, string> = {
  person: "Person",
  organization: "Organization",
  product: "Product",
};

export function OutlinePanel({
  outline,
  segments,
  capability,
  onJump,
}: OutlinePanelProps): React.JSX.Element | null {
  const canSeek = capability.seekEnabled;

  const jump = useCallback(
    (startSeconds: number) => {
      const line = resolveJumpIndex(segments, startSeconds);
      // -1 means the time is past the last rendered line; do nothing rather than
      // seek somewhere misleading.
      if (line >= 0) onJump(line);
    },
    [segments, onJump],
  );

  // When the feature is gated off, render nothing rather than nag a persistent
  // "turned off" panel onto every review page. The honest empty state is only
  // for when the feature IS on but no asset was generated yet.
  if (!outline || (!outline.available && outline.gated)) {
    return null;
  }

  if (!outline.available) {
    return (
      <section className="outline-panel my-2" aria-label="Outline">
        <h2>Outline</h2>
        <p className="muted">No outline was generated for this transcript.</p>
      </section>
    );
  }

  const dropped = totalDropped(outline.diagnostics);
  const hasContext =
    outline.context.summary !== null || outline.context.topics.length > 0;
  const entityCount = outline.mentions.length;

  return (
    <details className="outline-panel my-2" aria-label="Outline">
      <summary>
        Topics and entities{entityCount > 0 ? ` (${entityCount})` : ""}
      </summary>
      {outline.assetStale && (
        <p className="notice text-sm" role="note">
          This outline was built from an earlier version of the transcript, so
          the quoted text may have changed. Jumps still work. Regenerate it for an
          up-to-date outline.
        </p>
      )}

      {outline.mentions.length > 0 ? (
        <ul className="outline-entities">
          {outline.mentions.map((mention, mentionIndex) => (
            <li
              key={mentionIndex}
              className="outline-entity"
            >
              <p className="outline-entity-head">
                <span className="outline-surface">{mention.surface}</span>
                {mention.kind && (
                  <span className="outline-kind muted">
                    {" "}
                    · {KIND_LABEL[mention.kind] ?? mention.kind}
                  </span>
                )}
              </p>
              <ul className="outline-occurrences">
                {mention.occurrences.map((occ, index) => (
                  <li key={`${occ.segmentIndex}-${index}`}>
                    <button
                      type="button"
                      className="outline-jump"
                      onClick={() => jump(occ.startSeconds)}
                      title={
                        canSeek
                          ? "Play from here"
                          : "Go to this point in the transcript"
                      }
                    >
                      <span className="outline-time">
                        ≈ {formatClock(occ.startSeconds)}
                      </span>
                      {occ.quote && (
                        <span className="outline-quote muted">
                          {"“"}
                          {occ.quote}
                          {"”"}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      ) : dropped > 0 ? (
        <p className="muted">
          {dropped} mention{dropped === 1 ? "" : "s"} from an earlier version of the
          transcript could not be matched to the current text, so none are shown.
        </p>
      ) : (
        <p className="muted">
          No people, organizations, or products were found in this transcript.
        </p>
      )}

      {hasContext && (
        <div className="outline-context">
          <p className="muted text-sm">
            Summary and topics are context only. They are not linked to specific
            moments.
          </p>
          {outline.context.summary && (
            <p className="outline-summary">{outline.context.summary}</p>
          )}
          {outline.context.topics.length > 0 && (
            <ul className="outline-topics">
              {outline.context.topics.map((topic, index) => (
                <li key={index}>{topic}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {outline.mentions.length > 0 && dropped > 0 && (
        <p className="muted text-sm">
          {dropped} mention{dropped === 1 ? "" : "s"} could not be matched to the
          current transcript and {dropped === 1 ? "is" : "are"} not shown.
        </p>
      )}
    </details>
  );
}
