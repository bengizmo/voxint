import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

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
  // Called after a non-claim 409 (the action was re-ruled or the window ran
  // out), while the write guard is still held, so the caller can refetch
  // server truth before any other edit runs (issue #718).
  onConflict?: () => Promise<void>;
  // The editor's write guard. An undo holds it so no other edit can be in
  // flight at the same time and land its response out of order. `busy` disables
  // the button while another edit holds it, so a click there is never silently
  // dropped.
  writeGuard?: {
    busy: boolean;
    busyRef: RefObject<boolean>;
    setBusy: (busy: boolean) => void;
  };
}

const UNDO_COPY: Record<UndoPayload["kind"], string> = {
  enroll: "Enrollment applied.",
  decide: "Decision applied.",
  merge: "Labels merged.",
  relabel: "Segment speaker changed.",
};

// The undo route and its form fields for each kind. The nonce is derived from
// the undone action, so a retried click replays instead of undoing twice.
function undoRequest(undo: UndoPayload): [string, Record<string, string>] {
  if (undo.kind === "merge") {
    return [
      "merge",
      { merge_nonce: undo.mergeNonce, nonce: `undo:${undo.mergeNonce}` },
    ];
  }
  return [
    undo.kind,
    { decision_id: undo.decisionId, nonce: `undo:${undo.decisionId}` },
  ];
}

export function UndoToast({
  undo,
  runId,
  reviewToken,
  claimCsrf,
  onClaimLost,
  onUndone,
  onDismiss,
  onConflict,
  writeGuard,
}: UndoToastProps) {
  const localBusyRef = useRef(false);
  const busyRef = writeGuard?.busyRef ?? localBusyRef;
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
    if (busyRef.current || !claimCsrf) return;
    busyRef.current = true;
    setBusy(true);
    writeGuard?.setBusy(true);
    setError(null);
    try {
      const body = new URLSearchParams();
      body.append("token", reviewToken);
      body.append("csrf_token", claimCsrf);
      const [action, fields] = undoRequest(undo);
      for (const [key, value] of Object.entries(fields)) body.append(key, value);
      const res = await apiFetch(`/review/${runId}/undo/${action}`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: body.toString(),
      });
      onUndone((await res.json()) as LabelsResult);
    } catch (err) {
      if (err instanceof ApiError && err.conflictKind === "claim") {
        onClaimLost();
        onDismiss();
      } else if (err instanceof ApiError && err.status === 409) {
        const expired = Date.now() >= new Date(undo.expiresAt).getTime();
        setError(
          expired
            ? "Undo window expired."
            : "Too late to undo. This was changed again since.",
        );
        await onConflict?.();
      } else {
        setError(err instanceof ApiError ? err.detail : "Undo failed.");
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
      writeGuard?.setBusy(false);
    }
  }, [
    claimCsrf,
    reviewToken,
    undo,
    runId,
    onUndone,
    onClaimLost,
    onDismiss,
    onConflict,
    busyRef,
    writeGuard,
  ]);

  const label = UNDO_COPY[undo.kind];

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
          disabled={busy || (writeGuard?.busy ?? false)}
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
