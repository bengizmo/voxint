import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import { makeNonce } from "../lib/nonce";
import {
  coverage,
  headline,
  isAmbiguous,
  isConfirmable,
  isHumanRuling,
  partition,
  summary,
  whyText,
  type LabelStateShape,
} from "../lib/speaker-bands";
import type { Segment } from "./TranscriptPlayer";

export interface LabelsResult {
  labels: LabelStateShape[];
  segments: Segment[];
  progress: { verified: number; total: number };
  undo?: UndoPayload;
}

export type UndoPayload =
  | { kind: "enroll"; decisionId: string; expiresAt: string }
  | { kind: "merge"; mergeNonce: string; expiresAt: string };

interface SpeakerRailProps {
  runId: string;
  reviewToken: string | null;
  writable: boolean;
  labelStates: LabelStateShape[];
  speakers: { id: string; displayName: string }[];
  onClaimLost: () => void;
  onLabelsChanged: (result: LabelsResult) => void;
  onHearVoice?: (label: string) => void;
  hearableLabels?: ReadonlySet<string>;
}

type Speaker = { id: string; displayName: string };
type Decide = (label: string, action: string, speakerId?: string) => void;
type Enroll = (label: string, name: string) => void;

function RulingRow({
  state,
  speakers,
  busy,
  onDecide,
  onEnroll,
  mode,
}: {
  state: LabelStateShape;
  speakers: Speaker[];
  busy: boolean;
  onDecide: Decide;
  onEnroll: Enroll;
  mode: "needs-you" | "change";
}) {
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const confirmable = isConfirmable(state);
  const addPerson = () => {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    onEnroll(state.label, trimmed);
    setName("");
    setAdding(false);
  };
  const placeholder =
    mode === "change"
      ? "Reassign to…"
      : confirmable
        ? "Someone else…"
        : "Known person…";

  return (
    <>
      {mode === "needs-you" && confirmable && (
        <button
          type="button"
          className="primary"
          disabled={busy}
          onClick={() =>
            onDecide(state.label, "assign", state.candidateSpeakerId!)
          }
        >
          Confirm {state.candidateSpeakerName}
        </button>
      )}
      <div className="card-actions my-1">
        <select
          className="text-sm"
          value=""
          disabled={busy}
          aria-label={`${state.label}: choose who this is`}
          onChange={(event) => {
            const value = event.currentTarget.value;
            if (value === "__new") setAdding(true);
            else if (value) onDecide(state.label, "assign", value);
            event.currentTarget.value = "";
            event.currentTarget.blur();
          }}
        >
          <option value="">{placeholder}</option>
          {speakers.map((speaker) => (
            <option key={speaker.id} value={speaker.id}>
              {speaker.displayName}
            </option>
          ))}
          <option value="__new">Add a new person…</option>
        </select>
        <button
          type="button"
          className="text-sm secondary"
          title="Background noise, music, a TV: not a person to keep in the results."
          disabled={busy}
          onClick={() => onDecide(state.label, "exclude")}
        >
          Not a person
        </button>
        <button
          type="button"
          className="text-sm secondary"
          title="Records that you could not tell who this is. It settles this voice; you can change it later."
          disabled={busy}
          onClick={() => onDecide(state.label, "unknown")}
        >
          Can't tell
        </button>
      </div>
      {adding && (
        <div className="ruling-add flex items-center my-1">
          <input
            type="text"
            value={name}
            onChange={(event) => setName(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") addPerson();
            }}
            placeholder="new person's name"
            maxLength={120}
            className="text-sm mr-2"
            aria-label={`Name the new person for ${state.label}`}
            autoFocus
          />
          <button
            type="button"
            className="text-sm mr-2"
            disabled={busy || !name.trim()}
            onClick={addPerson}
          >
            Add person
          </button>
          <button
            type="button"
            className="text-sm secondary"
            disabled={busy}
            onClick={() => {
              setName("");
              setAdding(false);
            }}
          >
            Cancel
          </button>
        </div>
      )}
    </>
  );
}

function SpeakerCard({
  state,
  reviewToken,
  writable,
  speakers,
  busy,
  onDecide,
  onEnroll,
  onHearVoice,
}: {
  state: LabelStateShape;
  reviewToken: string | null;
  writable: boolean;
  speakers: Speaker[];
  busy: boolean;
  onDecide: Decide;
  onEnroll: Enroll;
  onHearVoice?: (label: string) => void;
}) {
  const copy = headline(state);
  const explanation = whyText(state);

  return (
    <div
      className={`label-card${state.paletteIndex != null ? ` spk-${state.paletteIndex}` : ""}`}
    >
      <h3>
        {state.label}{" "}
        <span
          className={`pill ${state.resolution === "auto_enroll" ? "grounded" : "unresolved"}`}
        >
          {state.resolution === "auto_enroll"
            ? "saved automatically"
            : "needs you"}
        </span>
      </h3>
      <p className="rail-title">{copy.title}</p>
      <p className="muted text-sm">{copy.detail}</p>
      <div className="flex items-center">
        <p className="muted text-sm">
          {state.turnCount} turns, {Math.round(state.totalSeconds)}s.
          {state.llmHintName && (
            <>
              {" "}
              Heard name (unverified): &ldquo;{state.llmHintName}&rdquo;.
            </>
          )}
        </p>
        {onHearVoice && (
          <button
            type="button"
            className="text-sm secondary"
            onClick={() => onHearVoice(state.label)}
          >
            Hear this voice
          </button>
        )}
      </div>
      {explanation && (
        <details className="match-why">
          <summary className="text-sm">
            {isAmbiguous(state)
              ? "Why no name?"
              : isConfirmable(state)
                ? "Why this match?"
                : "Why no match?"}
          </summary>
          <p className="muted text-sm">{explanation}</p>
        </details>
      )}
      {writable && reviewToken && (
        <RulingRow
          state={state}
          speakers={speakers}
          busy={busy}
          onDecide={onDecide}
          onEnroll={onEnroll}
          mode="needs-you"
        />
      )}
    </div>
  );
}

function ResolvedRow({
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
  speakers: Speaker[];
  busy: boolean;
  onDecide: Decide;
  onEnroll: Enroll;
}) {
  const [expanded, setExpanded] = useState(false);
  const copy = headline(state);
  const pill = isHumanRuling(state)
    ? "your ruling"
    : state.resolution === "auto_enroll"
      ? "saved automatically"
      : "voice match";

  return (
    <div
      className={`label-card rail-row${state.paletteIndex != null ? ` spk-${state.paletteIndex}` : ""}`}
    >
      <h3>
        {state.label}{" "}
        <span className={`pill ${isHumanRuling(state) ? "human" : "grounded"}`}>
          {pill}
        </span>
      </h3>
      <span className="rail-title">{copy.title}</span>
      <span className="muted text-sm">
        {copy.detail}{" "}
        {state.resolution === "auto_enroll" && !isConfirmable(state) && (
          <a href="/speakers" className="text-sm">
            Speakers page
          </a>
        )}
      </span>
      {writable && reviewToken && (
        <button
          type="button"
          className="rail-change text-sm secondary"
          aria-expanded={expanded}
          disabled={busy}
          onClick={() => setExpanded((open) => !open)}
        >
          {expanded ? "Hide" : "Change"}
        </button>
      )}
      {expanded && writable && reviewToken && (
        <RulingRow
          state={state}
          speakers={speakers}
          busy={busy}
          onDecide={onDecide}
          onEnroll={onEnroll}
          mode="change"
        />
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
              <option value="new">Add a new person…</option>
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
              placeholder="new person's name"
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
  onHearVoice,
  hearableLabels,
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

  useEffect(() => {
    setLabelStates(applyPalette(initialStates));
  }, [applyPalette, initialStates]);

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

  const groups = partition(labelStates);
  const settled = groups.needsYou.length + groups.tooShort.length === 0;
  const decideFromRow: Decide = (label, action, speakerId) => {
    void decide(label, action, speakerId);
  };
  const enrollFromRow: Enroll = (label, name) => {
    void enroll(label, name);
  };

  return (
    <div className="lib-sidebar" role="complementary" aria-label="Speaker rail">
      <p className="rail-summary" aria-live="polite">
        {summary(groups, coverage(labelStates))}
      </p>
      {groups.needsYou.length > 0 && (
        <section className="rail-group" aria-labelledby="rail-needs-you">
          <h3 id="rail-needs-you" className="rail-group-title">
            Needs you <span className="muted">({groups.needsYou.length})</span>
          </h3>
          {groups.needsYou.map((state) => (
            <SpeakerCard
              key={state.label}
              state={state}
              reviewToken={reviewToken}
              writable={writable}
              speakers={speakers}
              busy={busy}
              onDecide={decideFromRow}
              onEnroll={enrollFromRow}
              onHearVoice={
                onHearVoice && hearableLabels?.has(state.label)
                  ? onHearVoice
                  : undefined
              }
            />
          ))}
        </section>
      )}
      {groups.tooShort.length > 0 && (
        <details className="rail-group">
          <summary className="rail-group-title">
            Too little speech to identify{" "}
            <span className="muted">({groups.tooShort.length})</span>
          </summary>
          {groups.tooShort.map((state) => (
            <SpeakerCard
              key={state.label}
              state={state}
              reviewToken={reviewToken}
              writable={writable}
              speakers={speakers}
              busy={busy}
              onDecide={decideFromRow}
              onEnroll={enrollFromRow}
              onHearVoice={
                onHearVoice && hearableLabels?.has(state.label)
                  ? onHearVoice
                  : undefined
              }
            />
          ))}
        </details>
      )}
      {groups.matchedAutomatically.length > 0 && (
        <details className="rail-group" open={settled || undefined}>
          <summary className="rail-group-title">
            Matched automatically{" "}
            <span className="muted">
              ({groups.matchedAutomatically.length})
            </span>
          </summary>
          {groups.matchedAutomatically.map((state) => (
            <ResolvedRow
              key={state.label}
              state={state}
              reviewToken={reviewToken}
              writable={writable}
              speakers={speakers}
              busy={busy}
              onDecide={decideFromRow}
              onEnroll={enrollFromRow}
            />
          ))}
        </details>
      )}
      {groups.yourRulings.length > 0 && (
        <details className="rail-group" open={settled || undefined}>
          <summary className="rail-group-title">
            Your rulings{" "}
            <span className="muted">({groups.yourRulings.length})</span>
          </summary>
          {groups.yourRulings.map((state) => (
            <ResolvedRow
              key={state.label}
              state={state}
              reviewToken={reviewToken}
              writable={writable}
              speakers={speakers}
              busy={busy}
              onDecide={decideFromRow}
              onEnroll={enrollFromRow}
            />
          ))}
        </details>
      )}
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
