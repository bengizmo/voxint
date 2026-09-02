import { useCallback, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import { makeNonce } from "../lib/nonce";
import type { Segment } from "./TranscriptPlayer";

function confidenceBand(score: number | null): string {
  if (score == null || !isFinite(score)) return "unknown";
  if (score >= 0.8) return "likely";
  if (score >= 0.5) return "possible";
  return "low";
}

export interface LabelsResult {
  labels: LabelStateShape[];
  segments: Segment[];
  progress: { verified: number; total: number };
}

export interface LabelStateShape {
  label: string;
  paletteIndex: number | null;
  turnCount: number;
  totalSeconds: number;
  resolution: string;
  speakerId: string | null;
  speakerName: string | null;
  cosineConfidence: number | null;
  cosineSpeakerId: string | null;
  cosineSpeakerName: string | null;
  cosineGrounded: boolean;
  llmHintName: string | null;
  band: string | null;
  bandReason: string | null;
  candidatePromptAllowed: boolean;
  matchDecision: string | null;
  matchReason: string | null;
  matchMargin: number | null;
  matchEligibleSeconds: number;
}

interface SpeakerRailProps {
  runId: string;
  reviewToken: string | null;
  writable: boolean;
  labelStates: LabelStateShape[];
  speakers: { id: string; displayName: string }[];
  onClaimLost: () => void;
  onLabelsChanged: (result: LabelsResult) => void;
}

function isResolved(s: LabelStateShape): boolean {
  return s.resolution !== "unresolved";
}

function resolutionSummary(s: LabelStateShape): string {
  switch (s.resolution) {
    case "grounded_cosine":
      return `Machine-matched: ${s.speakerName}`;
    case "human_assign":
      return `Assigned: ${s.speakerName}`;
    case "auto_enroll":
      return `Auto-enrolled: ${s.speakerName}`;
    case "human_exclude":
      return "Excluded";
    case "human_unknown":
      return "Marked unknown";
    default:
      return s.resolution;
  }
}

function resolutionBadge(s: LabelStateShape): React.JSX.Element {
  switch (s.resolution) {
    case "unresolved":
      return <span className="pill unresolved">needs ruling</span>;
    case "grounded_cosine":
      return <span className="pill grounded">machine: {s.speakerName}</span>;
    case "human_assign":
      return <span className="pill human">assigned: {s.speakerName}</span>;
    case "auto_enroll":
      return <span className="pill grounded">auto: {s.speakerName}</span>;
    case "human_exclude":
      return <span className="pill human">excluded</span>;
    case "human_unknown":
      return <span className="pill human">unknown</span>;
    default:
      return <span className="pill">{s.resolution}</span>;
  }
}

function SpeakerCard({
  state,
  reviewToken,
  writable,
  speakers,
  busy,
  onDecide,
  onEnroll,
}: {
  state: LabelStateShape;
  reviewToken: string | null;
  writable: boolean;
  speakers: { id: string; displayName: string }[];
  busy: boolean;
  onDecide: (label: string, action: string, speakerId?: string) => void;
  onEnroll: (label: string, name: string) => void;
}) {
  const [enrollName, setEnrollName] = useState("");
  const [expanded, setExpanded] = useState(false);
  const resolved = isResolved(state);

  return (
    <div
      className={`label-card${state.paletteIndex != null ? ` spk-${state.paletteIndex}` : ""}`}
    >
      {resolved ? (
        <>
          <h3 className="flex items-center">
            {state.label}{" "}
            <span className="muted text-sm ml-2">{resolutionSummary(state)}</span>
            {writable && reviewToken && (
              <button
                type="button"
                onClick={() => setExpanded((on) => !on)}
                aria-expanded={expanded}
                className="text-sm ml-auto secondary"
              >
                {expanded ? "Hide" : "Change"}
              </button>
            )}
          </h3>
          {expanded && writable && reviewToken && (
            <div className="card-actions my-1">
              <div className="flex items-center my-1">
                <select
                  className="text-sm mr-2"
                  value=""
                  disabled={busy}
                  aria-label={`Reassign ${state.label} to a different speaker`}
                  onChange={(e) => {
                    const val = e.target.value;
                    e.currentTarget.blur();
                    if (val) onDecide(state.label, "assign", val);
                  }}
                >
                  <option value="">Reassign to…</option>
                  {speakers.map((sp) => (
                    <option key={sp.id} value={sp.id}>
                      {sp.displayName}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => onDecide(state.label, "exclude")}
                  disabled={busy}
                  className="text-sm mr-2 secondary"
                >
                  Exclude
                </button>
                <button
                  type="button"
                  onClick={() => onDecide(state.label, "unknown")}
                  disabled={busy}
                  className="text-sm secondary"
                >
                  Unknown
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <h3>
            {state.label} {resolutionBadge(state)}
          </h3>
          <p className="muted text-sm">
            {state.turnCount} turns, {Math.round(state.totalSeconds)}s.
            {state.cosineSpeakerName && (
              <>
                {" "}
                {state.cosineGrounded ? "Strong" : "Possible"} voice match:{" "}
                {state.cosineSpeakerName}.
              </>
            )}
            {state.llmHintName && (
              <>
                {" "}
                Heard name (unverified): &ldquo;{state.llmHintName}&rdquo;.
              </>
            )}
          </p>
          {state.cosineSpeakerName && (
            <details className="match-why">
              <summary className="text-sm">Why this match?</summary>
              <p className="muted text-sm">
                Voice similarity {confidenceBand(state.cosineConfidence)} ({state.cosineConfidence?.toFixed(2) ?? "—"}){" "}
                to {state.cosineSpeakerName}
                {state.cosineGrounded
                  ? ", strong enough to trust on its own"
                  : ", not strong enough to confirm without your check"}
                .
              </p>
            </details>
          )}
          {writable && reviewToken && (
            <div className="card-actions my-1">
              {state.cosineGrounded && state.cosineSpeakerId ? (
                <>
                  <button
                    type="button"
                    onClick={() =>
                      onDecide(state.label, "assign", state.cosineSpeakerId!)
                    }
                    disabled={busy}
                    className="primary mr-2"
                  >
                    Confirm {state.cosineSpeakerName}
                  </button>
                  <details>
                    <summary className="text-sm secondary">More options</summary>
                    <div className="flex items-center my-1">
                      <select
                        className="text-sm mr-2"
                        value=""
                        disabled={busy}
                        aria-label={`Assign ${state.label} to a speaker`}
                        onChange={(e) => {
                          const val = e.target.value;
                          e.currentTarget.blur();
                          if (val) onDecide(state.label, "assign", val);
                        }}
                      >
                        <option value="">Assign to…</option>
                        {speakers.map((sp) => (
                          <option key={sp.id} value={sp.id}>
                            {sp.displayName}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        onClick={() => onDecide(state.label, "exclude")}
                        disabled={busy}
                        className="text-sm mr-2 secondary"
                      >
                        Exclude
                      </button>
                      <button
                        type="button"
                        onClick={() => onDecide(state.label, "unknown")}
                        disabled={busy}
                        className="text-sm secondary"
                      >
                        Unknown
                      </button>
                    </div>
                    <div className="flex items-center my-1">
                      <input
                        type="text"
                        value={enrollName}
                        onChange={(e) => setEnrollName(e.target.value)}
                        placeholder="new speaker name"
                        maxLength={120}
                        className="text-sm mr-2"
                        aria-label={`Enroll ${state.label} as a new speaker`}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const name = enrollName.trim();
                          if (name) {
                            onEnroll(state.label, name);
                            setEnrollName("");
                          }
                        }}
                        disabled={busy || !enrollName.trim()}
                        className="text-sm"
                      >
                        Enroll new
                      </button>
                    </div>
                  </details>
                </>
              ) : (
                <>
                  <p className="text-sm">Who is this?</p>
                  <div className="flex items-center my-1">
                    <select
                      className="text-sm mr-2"
                      value=""
                      disabled={busy}
                      aria-label={`Assign ${state.label} to a known speaker`}
                      onChange={(e) => {
                        const val = e.target.value;
                        e.currentTarget.blur();
                        if (val) onDecide(state.label, "assign", val);
                      }}
                    >
                      <option value="">Known person…</option>
                      {speakers.map((sp) => (
                        <option key={sp.id} value={sp.id}>
                          {sp.displayName}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => onDecide(state.label, "exclude")}
                      disabled={busy}
                      className="text-sm mr-2 secondary"
                      title="Mark this speaker label as not a person (background noise, music, etc.)"
                    >
                      Not a person
                    </button>
                    <button
                      type="button"
                      onClick={() => onDecide(state.label, "unknown")}
                      disabled={busy}
                      className="text-sm secondary"
                      title="Skip for now — come back to this label later"
                    >
                      Not sure
                    </button>
                  </div>
                  <div className="flex items-center my-1">
                    <input
                      type="text"
                      value={enrollName}
                      onChange={(e) => setEnrollName(e.target.value)}
                      placeholder="new speaker name"
                      maxLength={120}
                      className="text-sm mr-2"
                      aria-label={`Enroll ${state.label} as a new speaker`}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const name = enrollName.trim();
                        if (name) {
                          onEnroll(state.label, name);
                          setEnrollName("");
                        }
                      }}
                      disabled={busy || !enrollName.trim()}
                      className="text-sm"
                    >
                      Add person
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

interface MergePreview {
  labels: string[];
  speakerId: string | null;
  speakerName: string;
  turnsMoved: number;
  expected: Record<string, string | null>;
}

function MergePanel({
  runId,
  reviewToken,
  labelStates,
  speakers,
  busy,
  onMerge,
  onClaimLost,
}: {
  runId: string;
  reviewToken: string | null;
  labelStates: LabelStateShape[];
  speakers: { id: string; displayName: string }[];
  busy: boolean;
  onMerge: (data: LabelsResult) => void;
  onClaimLost: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [target, setTarget] = useState("");
  const [newName, setNewName] = useState("");
  const [preview, setPreview] = useState<MergePreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [mergeBusy, setMergeBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleLabel = (label: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
    setPreview(null);
    setError(null);
  };

  const doPreview = async () => {
    if (!reviewToken || selected.size < 2 || !target) return;
    setPreviewBusy(true);
    setError(null);
    try {
      const body = new URLSearchParams();
      body.append("token", reviewToken);
      body.append("target", target);
      if (target === "new" && newName.trim()) {
        body.append("new_name", newName.trim());
      }
      for (const l of selected) body.append("labels", l);
      const res = await apiFetch(`/review/${runId}/merge/preview`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: body.toString(),
      });
      const data = (await res.json()) as MergePreview;
      setPreview(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        onClaimLost();
      } else {
        setError(err instanceof ApiError ? err.detail : "Preview failed.");
      }
    } finally {
      setPreviewBusy(false);
    }
  };

  const doMerge = async () => {
    if (!reviewToken || !preview) return;
    setMergeBusy(true);
    setError(null);
    try {
      const body = new URLSearchParams();
      body.append("token", reviewToken);
      body.append("nonce", makeNonce());
      body.append("expected", JSON.stringify(preview.expected));
      if (preview.speakerId) body.append("speaker_id", preview.speakerId);
      else if (newName.trim()) body.append("display_name", newName.trim());
      for (const l of preview.labels) body.append("labels", l);
      const res = await apiFetch(`/review/${runId}/merge`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: body.toString(),
      });
      const data = (await res.json()) as LabelsResult;
      setPreview(null);
      setSelected(new Set());
      setTarget("");
      setNewName("");
      onMerge(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        onClaimLost();
      } else {
        setError(err instanceof ApiError ? err.detail : "Merge failed.");
      }
    } finally {
      setMergeBusy(false);
    }
  };

  if (labelStates.length < 2) return null;

  return (
    <div className="card my-2">
      <details>
        <summary>
          <h3 style={{ display: "inline" }}>Same speaker across labels?</h3>
        </summary>
        <p className="muted text-sm my-1">
          Tick labels that are the same speaker, choose who they are, then
          preview before applying.
        </p>
        <fieldset className="merge-labels">
          <legend className="text-sm">Same speaker:</legend>
          {labelStates.map((s) => (
            <label
              key={s.label}
              className={`text-sm${s.paletteIndex != null ? ` spk-${s.paletteIndex}` : ""}`}
            >
              <input
                type="checkbox"
                checked={selected.has(s.label)}
                disabled={busy || mergeBusy}
                onChange={() => toggleLabel(s.label)}
              />{" "}
              {s.label}
              <span className="muted">
                {" "}
                ({s.turnCount} turns
                {s.speakerName ? `, ${s.speakerName}` : ""})
              </span>
            </label>
          ))}
        </fieldset>
        <div className="flex items-center my-1">
          <label className="text-sm mr-2">
            Who are they?{" "}
            <select
              value={target}
              onChange={(e) => {
                setTarget(e.target.value);
                setPreview(null);
              }}
              className="text-sm"
              disabled={busy || mergeBusy}
            >
              <option value="">choose</option>
              <option value="new">Enroll a new speaker…</option>
              {speakers.map((sp) => (
                <option key={sp.id} value={sp.id}>
                  {sp.displayName}
                </option>
              ))}
            </select>
          </label>
          {target === "new" && (
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="new speaker name"
              maxLength={120}
              className="text-sm mr-2"
            />
          )}
        </div>
        <div className="my-1">
          <button
            type="button"
            onClick={() => void doPreview()}
            disabled={
              selected.size < 2 ||
              !target ||
              (target === "new" && !newName.trim()) ||
              previewBusy ||
              mergeBusy
            }
            className="text-sm mr-2"
          >
            {previewBusy ? "Loading…" : "Preview merge…"}
          </button>
        </div>
        {preview && (
          <div className="my-1" aria-live="polite">
            <p className="text-sm">
              Merge {preview.labels.join(", ")} → {preview.speakerName} (
              {preview.turnsMoved} turns moved).
            </p>
            <button
              type="button"
              onClick={() => void doMerge()}
              disabled={mergeBusy}
              className="primary text-sm mr-2"
            >
              {mergeBusy ? "Merging…" : "Confirm merge"}
            </button>
            <button
              type="button"
              onClick={() => setPreview(null)}
              disabled={mergeBusy}
              className="text-sm secondary"
            >
              Cancel
            </button>
          </div>
        )}
        {error && (
          <p role="alert" className="text-sm" style={{ color: "var(--error)" }}>
            {error}
          </p>
        )}
      </details>
    </div>
  );
}

export function SpeakerRail({
  runId,
  reviewToken,
  writable,
  labelStates: initialStates,
  speakers,
  onClaimLost,
  onLabelsChanged,
}: SpeakerRailProps) {
  const [labelStates, setLabelStates] =
    useState<LabelStateShape[]>(initialStates);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);

  const paletteMap = useRef(
    new Map(initialStates.map((s) => [s.label, s.paletteIndex])),
  );

  const applyPalette = useCallback(
    (states: LabelStateShape[]): LabelStateShape[] =>
      states.map((s) => ({
        ...s,
        paletteIndex: s.paletteIndex ?? paletteMap.current.get(s.label) ?? null,
      })),
    [],
  );

  const adoptResult = useCallback(
    (data: LabelsResult) => {
      const enriched = applyPalette(data.labels);
      setLabelStates(enriched);
      onLabelsChanged({ ...data, labels: enriched });
    },
    [applyPalette, onLabelsChanged],
  );

  const decide = useCallback(
    async (label: string, action: string, speakerId?: string) => {
      if (!reviewToken || busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const body: Record<string, string> = {
          token: reviewToken,
          nonce: makeNonce(),
          action,
        };
        if (speakerId) body.speaker_id = speakerId;
        const res = await apiFetch(
          `/review/${runId}/labels/${encodeURIComponent(label)}/decision`,
          {
            method: "POST",
            headers: {
              "content-type": "application/x-www-form-urlencoded",
              accept: "application/json",
            },
            body: new URLSearchParams(body).toString(),
          },
        );
        const data = (await res.json()) as LabelsResult;
        adoptResult(data);
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          onClaimLost();
        } else {
          setError(err instanceof ApiError ? err.detail : "Decision failed.");
        }
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [reviewToken, runId, onClaimLost, adoptResult],
  );

  const enroll = useCallback(
    async (label: string, displayName: string) => {
      if (!reviewToken || busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const body: Record<string, string> = {
          token: reviewToken,
          nonce: makeNonce(),
          display_name: displayName,
        };
        const res = await apiFetch(
          `/review/${runId}/labels/${encodeURIComponent(label)}/enroll`,
          {
            method: "POST",
            headers: {
              "content-type": "application/x-www-form-urlencoded",
              accept: "application/json",
            },
            body: new URLSearchParams(body).toString(),
          },
        );
        const data = (await res.json()) as LabelsResult;
        adoptResult(data);
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          onClaimLost();
        } else {
          setError(err instanceof ApiError ? err.detail : "Enrollment failed.");
        }
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [reviewToken, runId, onClaimLost, adoptResult],
  );

  // Sort: unresolved labels first, then resolved.
  const sortedStates = [...labelStates].sort((a, b) => {
    const aR = isResolved(a) ? 1 : 0;
    const bR = isResolved(b) ? 1 : 0;
    return aR - bR;
  });

  return (
    <div className="lib-sidebar" role="complementary" aria-label="Speaker rail">
      {sortedStates.map((s) => (
        <SpeakerCard
          key={s.label}
          state={s}
          reviewToken={reviewToken}
          writable={writable}
          speakers={speakers}
          busy={busy}
          onDecide={(l, a, sp) => void decide(l, a, sp)}
          onEnroll={(l, n) => void enroll(l, n)}
        />
      ))}
      {writable && reviewToken && (
        <MergePanel
          runId={runId}
          reviewToken={reviewToken}
          labelStates={labelStates}
          speakers={speakers}
          busy={busy}
          onMerge={adoptResult}
          onClaimLost={onClaimLost}
        />
      )}
      {error && (
        <p role="alert" className="text-sm" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
