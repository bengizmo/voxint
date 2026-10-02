import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import type { WriteGuard } from "../lib/editor-mutations";
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
  // Called after a non-claim 409 (the action was re-ruled, the window ran
  // out, or the speaker it would restore is archived), while the write guard is still held, so the caller can refetch
  // server truth before any other edit runs (issue #718). It must settle
  // promptly and never reject: the guard stays held until it does.
  onConflict?: () => Promise<void>;
  // Shared by the editor, rail, and merge suggestion. Undo holds it through
  // conflict refetches so whole-run responses cannot land out of order.
  // `busy` disables the button while another edit holds it.
  writeGuard?: WriteGuard;
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
  // The undo was refused only because the speaker it would restore is archived.
  // Restoring that speaker lets the same undo succeed, so the button stays.
  const [retryable, setRetryable] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-dismiss at the deadline, but not once the toast shows why an undo
  // failed: that stays until the operator closes it (issue #718). A retryable
  // failure still watches the deadline, and past it says the window closed and
  // drops the Undo button rather than offer an undo that can only fail.
  useEffect(() => {
    if (error !== null && !retryable) return;
    const expire = retryable
      ? () => {
          setError("Undo window expired.");
          setRetryable(false);
        }
      : onDismiss;
    const ms = new Date(undo.expiresAt).getTime() - Date.now();
    if (ms <= 0) {
      expire();
      return;
    }
    timerRef.current = setTimeout(expire, ms);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [undo.expiresAt, onDismiss, error, retryable]);

  const doUndo = useCallback(async () => {
    if (busyRef.current || !claimCsrf) return;
    busyRef.current = true;
    setBusy(true);
    writeGuard?.setBusy(true);
    setError(null);
    setRetryable(false);
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
      } else if (err instanceof ApiError && err.conflictKind === "archived-speaker") {
        // The server names the speaker to restore, which for a merged speaker
        // is the one it was merged into.
        setError(`Can't undo: ${err.detail}`);
        setRetryable(true);
        await onConflict?.();
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
      {(!error || retryable) && (
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
