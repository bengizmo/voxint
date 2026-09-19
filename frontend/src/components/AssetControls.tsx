import { useCallback, useEffect, useState } from "react";

import { apiFetch, ApiError } from "../lib/api-client";
import type { PollableAssetState } from "../lib/enrichment-polling";

export interface AssetControlsProps {
  runId: string;
  gatesOpen: boolean;
  sourceProblem: string | null;
  anyActive: boolean;
  kinds: {
    kind: string;
    title: string;
    hasAsset: boolean;
    stale: boolean;
    jobActive: boolean;
    jobId: string | null;
    jobStatus: string | null;
  }[];
  csrfGenerate: string;
  csrfCancel: string;
  polledState?: PollableAssetState | null;
  onActive?: () => void;
}

type AssetPhase = "idle" | "generating" | "generated" | "error";

export function AssetControls({
  runId,
  gatesOpen,
  sourceProblem,
  anyActive,
  kinds,
  csrfGenerate,
  csrfCancel,
  polledState,
  onActive,
}: AssetControlsProps): React.JSX.Element {
  const [phase, setPhase] = useState<AssetPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [retryKind, setRetryKind] = useState<string | undefined>();
  const [cancelStates, setCancelStates] = useState<
    Record<string, "pending" | "requested">
  >({});
  const [cancelErrors, setCancelErrors] = useState<Record<string, string>>({});

  const displayKinds = polledState?.kinds ?? kinds;
  const displayActive = polledState
    ? polledState.anyActive
    : anyActive || kinds.some((entry) => entry.jobActive);

  useEffect(() => {
    if (phase === "generated" && polledState && !polledState.anyActive) {
      setPhase("idle");
    }
  }, [phase, polledState]);

  const generate = useCallback(
    async (kind?: string) => {
      if (!gatesOpen || sourceProblem !== null || phase === "generating")
        return;
      setPhase("generating");
      setError(null);
      setRetryKind(kind);
      try {
        const body = new URLSearchParams({ csrf_token: csrfGenerate });
        if (kind) body.set("kind", kind);
        const response = await apiFetch(`/runs/${runId}/assets/generate`, {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body,
        });
        const result = (await response.json()) as {
          started: boolean;
          error: string | null;
          created: number;
        };
        if (result.error || !result.started || result.created === 0) {
          setError(
            result.error ||
              "No generation jobs started. Reload to check active jobs, then retry.",
          );
          setPhase("error");
        } else {
          setPhase("generated");
          onActive?.();
        }
      } catch (err) {
        setError(
          err instanceof ApiError ? err.detail : "Generation could not start.",
        );
        setPhase("error");
      }
    },
    [runId, csrfGenerate, gatesOpen, sourceProblem, phase, onActive],
  );

  const cancel = useCallback(
    async (jobId: string) => {
      if (cancelStates[jobId]) return;
      setCancelStates((states) => ({ ...states, [jobId]: "pending" }));
      setCancelErrors((errors) => ({ ...errors, [jobId]: "" }));
      try {
        const response = await apiFetch(
          `/runs/${runId}/assets/${jobId}/cancel`,
          {
            method: "POST",
            headers: {
              Accept: "application/json",
              "Content-Type": "application/x-www-form-urlencoded",
            },
            body: new URLSearchParams({ csrf_token: csrfCancel }),
          },
        );
        const result = (await response.json()) as { cancelled: boolean };
        if (!result.cancelled)
          throw new Error("Cancellation was not accepted.");
        setCancelStates((states) => ({ ...states, [jobId]: "requested" }));
      } catch (err) {
        setCancelErrors((errors) => ({
          ...errors,
          [jobId]:
            err instanceof ApiError
              ? err.detail
              : "Cancellation failed. Try again.",
        }));
        setCancelStates((states) => {
          const next = { ...states };
          delete next[jobId];
          return next;
        });
      }
    },
    [runId, csrfCancel, cancelStates],
  );

  if (!gatesOpen)
    return (
      <p className="muted text-sm">
        Run assets are off. Enable in Settings → Features.
      </p>
    );
  if (sourceProblem !== null)
    return <p className="notice text-sm">{sourceProblem}</p>;

  const allHaveAssets =
    displayKinds.length > 0 && displayKinds.every((entry) => entry.hasAsset);

  return (
    <section className="text-sm" aria-label="Asset controls">
      {phase === "idle" && (
        <button
          type="button"
          disabled={displayActive || displayKinds.length === 0}
          onClick={() => void generate()}
        >
          {allHaveAssets ? "Regenerate all" : "Generate all"}
        </button>
      )}
      {(phase === "generating" || (phase === "idle" && displayActive)) && (
        <p className="muted" role="status">
          Generating...
        </p>
      )}
      {phase === "generated" && (
        <p className="notice" role="status">
          {displayActive
            ? "Generating..."
            : "Generated. Reload to see updated content."}
        </p>
      )}
      {phase === "error" && (
        <p className="notice" role="alert">
          {error}{" "}
          <button type="button" onClick={() => void generate(retryKind)}>
            Retry
          </button>
        </p>
      )}
      {displayKinds.map((entry) =>
        entry.jobActive ? (
          <p key={entry.kind}>
            {entry.title}: {entry.jobStatus || "Generating..."}{" "}
            {entry.jobId && (
              <>
                <button
                  type="button"
                  className="inline"
                  disabled={!!cancelStates[entry.jobId]}
                  onClick={() => void cancel(entry.jobId!)}
                  aria-label={`Cancel ${entry.title}`}
                >
                  {cancelStates[entry.jobId] === "pending"
                    ? "Cancelling..."
                    : "Cancel"}
                </button>
                {cancelStates[entry.jobId] === "requested" && (
                  <span className="muted" role="status">
                    {" "}
                    Cancellation requested. Reload to check status.
                  </span>
                )}
                {cancelErrors[entry.jobId] && (
                  <span className="notice" role="alert">
                    {" "}
                    {cancelErrors[entry.jobId]}
                  </span>
                )}
              </>
            )}
          </p>
        ) : entry.hasAsset && entry.stale && phase === "idle" ? (
          <p key={entry.kind}>
            {entry.title}:{" "}
            <button
              type="button"
              className="inline"
              onClick={() => void generate(entry.kind)}
              aria-label={`Regenerate ${entry.title}`}
            >
              Regenerate
            </button>
          </p>
        ) : null,
      )}
    </section>
  );
}
