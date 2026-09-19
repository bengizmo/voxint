import { useEffect, useRef, useState } from "react";

import { apiFetch } from "../lib/api-client";

export interface PollableAssetState {
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
}

export interface PollableTranslateState {
  active: boolean;
  hasTranslation: boolean;
  activeJobId: string | null;
  jobStatus: string | null;
}

interface UseEnrichmentPollingOptions {
  runId: string;
  pollAssets: boolean;
  pollTranslate: boolean;
  intervalMs?: number;
}

interface UseEnrichmentPollingResult {
  assetState: PollableAssetState | null;
  translateState: PollableTranslateState | null;
}

function usePollingState<T>(
  path: string,
  enabled: boolean,
  intervalMs: number,
): T | null {
  const [state, setState] = useState<T | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!enabled) {
      setState(null);
      return;
    }

    const controller = new AbortController();
    controllerRef.current = controller;
    let inFlight = false;

    const poll = async () => {
      // Slow requests must not overlap and overwrite newer responses.
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await apiFetch(path, {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        const nextState = (await response.json()) as T;
        if (!controller.signal.aborted) setState(nextState);
      } catch {
        // A later interval retries transient failures; aborts are harmless.
      } finally {
        inFlight = false;
      }
    };

    void poll();
    const interval = setInterval(() => void poll(), intervalMs);
    return () => {
      clearInterval(interval);
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [path, enabled, intervalMs]);

  return state;
}

export function useEnrichmentPolling({
  runId,
  pollAssets,
  pollTranslate,
  intervalMs = 4000,
}: UseEnrichmentPollingOptions): UseEnrichmentPollingResult {
  const assetState = usePollingState<PollableAssetState>(
    `/runs/${runId}/assets`,
    pollAssets,
    intervalMs,
  );
  const translateState = usePollingState<PollableTranslateState>(
    `/runs/${runId}/translation`,
    pollTranslate,
    intervalMs,
  );

  return { assetState, translateState };
}
