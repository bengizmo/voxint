import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import type { LabelsResult } from "./SpeakerRail";

type UndoPayload = NonNullable<LabelsResult["undo"]>;

interface UndoToastProps {
  undo: UndoPayload;
  runId: string;
  reviewToken: string;
  claimCsrf: string | null;
  onClaimLost: () => void;
  onUndone: (data: LabelsResult) => void;
  onDismiss: () => void;
}

export function UndoToast({
  undo,
  runId,
  reviewToken,
  claimCsrf,
  onClaimLost,
  onUndone,
  onDismiss,
}: UndoToastProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const ms = new Date(undo.expiresAt).getTime() - Date.now();
    if (ms <= 0) {
      onDismiss();
      return;
    }
    timerRef.current = setTimeout(onDismiss, ms);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [undo.expiresAt, onDismiss]);

  const doUndo = useCallback(async () => {
    if (busy || !claimCsrf) return;
    setBusy(true);
    setError(null);
    try {
      const body = new URLSearchParams();
      body.append("token", reviewToken);
      body.append("csrf_token", claimCsrf);
      if (undo.kind === "enroll") {
        body.append("decision_id", undo.decisionId);
        body.append("nonce", `undo:${undo.decisionId}`);
        const res = await apiFetch(`/review/${runId}/undo/enroll`, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
          body: body.toString(),
        });
        onUndone((await res.json()) as LabelsResult);
      } else {
        body.append("merge_nonce", undo.mergeNonce);
        body.append("nonce", `undo:${undo.mergeNonce}`);
        const res = await apiFetch(`/review/${runId}/undo/merge`, {
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
          body: body.toString(),
        });
        onUndone((await res.json()) as LabelsResult);
      }
    } catch (err) {
      if (err instanceof ApiError && err.conflictKind === "claim") {
        onClaimLost();
        onDismiss();
      } else if (err instanceof ApiError && err.status === 409) {
        setError("Too late to undo — the attribution was changed since.");
      } else {
        setError(err instanceof ApiError ? err.detail : "Undo failed.");
      }
    } finally {
      setBusy(false);
    }
  }, [busy, claimCsrf, reviewToken, undo, runId, onUndone, onClaimLost, onDismiss]);

  const label =
    undo.kind === "enroll" ? "Enrollment applied." : "Labels merged.";

  return (
    <div
      style={{
        position: "fixed",
        bottom: "var(--space-4, 1rem)",
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
        maxWidth: "min(22rem, calc(100vw - 2rem))",
        animation: "toast-in 0.18s ease-out",
      }}
      role="status"
      aria-live="polite"
    >
      <span style={{ flex: 1, fontSize: "0.875rem" }}>
        {error ?? label}
      </span>
      {!error && (
        <button
          type="button"
          onClick={doUndo}
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
          {busy ? "Undoing…" : "Undo"}
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
