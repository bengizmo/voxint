import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import type { MergeSuggestion } from "../lib/merge-candidates";
import { makeNonce } from "../lib/nonce";
import type { LabelsResult } from "./SpeakerRail";

interface MergePreview {
  labels: string[];
  speakerId: string | null;
  speakerName: string;
  turnsMoved: number;
  expected: Record<string, string | null>;
}

interface MergeSuggestionToastProps {
  suggestion: MergeSuggestion;
  runId: string;
  reviewToken: string;
  onClaimLost: () => void;
  onMerged: (data: LabelsResult) => void;
  onDismiss: () => void;
  stacked?: boolean;
}

const AUTO_DISMISS_MS = 15_000;

export function MergeSuggestionToast({
  suggestion,
  runId,
  reviewToken,
  onClaimLost,
  onMerged,
  onDismiss,
  stacked = false,
}: MergeSuggestionToastProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  useEffect(() => {
    timerRef.current = setTimeout(() => onDismissRef.current(), AUTO_DISMISS_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const doMerge = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    if (timerRef.current) clearTimeout(timerRef.current);
    try {
      const allLabels = [suggestion.assignedLabel, ...suggestion.candidateLabels];
      const previewBody = new URLSearchParams();
      previewBody.append("token", reviewToken);
      previewBody.append("target", suggestion.targetSpeakerId);
      for (const l of allLabels) previewBody.append("labels", l);
      const previewRes = await apiFetch(`/review/${runId}/merge/preview`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: previewBody.toString(),
      });
      const preview = (await previewRes.json()) as MergePreview;

      const mergeBody = new URLSearchParams();
      mergeBody.append("token", reviewToken);
      mergeBody.append("nonce", makeNonce());
      mergeBody.append("expected", JSON.stringify(preview.expected));
      if (preview.speakerId) mergeBody.append("speaker_id", preview.speakerId);
      for (const l of preview.labels) mergeBody.append("labels", l);
      const mergeRes = await apiFetch(`/review/${runId}/merge`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: mergeBody.toString(),
      });
      const data = (await mergeRes.json()) as LabelsResult;
      onMerged(data);
    } catch (err) {
      if (err instanceof ApiError && err.conflictKind === "claim") {
        onClaimLost();
        onDismiss();
      } else if (err instanceof ApiError && err.status === 409) {
        setError("Labels changed since the suggestion was shown.");
        onDismiss();
      } else {
        setError(err instanceof ApiError ? err.detail : "Merge failed.");
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [suggestion, reviewToken, runId, onMerged, onClaimLost, onDismiss]);

  const candidateNames = suggestion.candidateLabels.join(", ");

  return (
    <div
      style={{
        position: "fixed",
        bottom: stacked ? "calc(var(--space-4, 1rem) + 4rem)" : "var(--space-4, 1rem)",
        right: "var(--space-4, 1rem)",
        background: "var(--paper, #fff)",
        color: "var(--ink, #222)",
        border: "1px solid var(--line, #ccc)",
        borderRadius: "var(--r-md, 0.5rem)",
        boxShadow: "var(--shadow-1, 0 1px 3px rgba(0,0,0,0.12))",
        padding: "0.75rem 1rem",
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        zIndex: 1200,
        maxWidth: "min(28rem, calc(100vw - 2rem))",
        animation: "toast-in 0.18s ease-out",
      }}
      role="status"
      aria-live="polite"
    >
      <span style={{ flex: 1, fontSize: "0.875rem" }}>
        {error ?? `${candidateNames} also ${suggestion.candidateLabels.length === 1 ? "sounds" : "sound"} like ${suggestion.targetSpeakerName}. Merge?`}
      </span>
      {!error && (
        <button
          type="button"
          onClick={() => void doMerge()}
          disabled={busy}
          style={{
            background: "none",
            border: "none",
            color: "var(--accent, teal)",
            fontWeight: 600,
            cursor: busy ? "wait" : "pointer",
            fontSize: "0.875rem",
            padding: "0.25rem 0.5rem",
            whiteSpace: "nowrap",
          }}
        >
          {busy ? "Merging…" : `Merge${suggestion.candidateLabels.length > 1 ? " all" : ""}`}
        </button>
      )}
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        style={{
          background: "none",
          border: "none",
          color: "var(--ink-muted, #888)",
          cursor: "pointer",
          fontSize: "1rem",
          lineHeight: 1,
          padding: "0.125rem",
        }}
      >
        ×
      </button>
    </div>
  );
}
