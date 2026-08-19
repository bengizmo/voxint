import { useRef, useState } from "react";

// Console corrections editor (issue #84). Lets a non-technical operator author
// literal find→replace correction rules from Settings instead of hand-editing a
// domain pack's manifest.yaml. The editor owns add/edit/remove/reorder
// client-side and submits the FULL ordered list (replace-all) to
// POST /settings/corrections, which validates through the SAME #80 gate a pack
// gets and is the AUTHORITATIVE check — the client-side checks here only spare an
// obviously-doomed round trip and surface the server's per-row message inline.

export interface CorrectionRuleProps {
  id: string;
  match: string;
  replace: string;
  case_sensitive: boolean;
  whole_word: boolean;
}

interface Limits {
  maxRules: number;
  maxMatchChars: number;
  maxReplacementChars: number;
}

export interface CorrectionsEditorProps {
  rules: CorrectionRuleProps[];
  action: string;
  csrfToken: string;
  limits: Limits;
}

interface Row extends CorrectionRuleProps {
  key: number;
}

interface FieldError {
  message: string;
  // 0-based row the server (or a client check) flagged; null = whole-list fault.
  row: number | null;
}

interface SaveEnvelope {
  ok: boolean;
  error?: string;
  row?: number | null;
  corrections?: CorrectionRuleProps[];
}

function toRow(rule: CorrectionRuleProps, key: number): Row {
  return {
    key,
    id: rule.id ?? "",
    match: rule.match ?? "",
    replace: rule.replace ?? "",
    // Both flags default true — the conservative posture from the #80 schema.
    case_sensitive: rule.case_sensitive ?? true,
    whole_word: rule.whole_word ?? true,
  };
}

export function CorrectionsEditor(props: CorrectionsEditorProps) {
  const { action, csrfToken, limits } = props;
  const nextKey = useRef(0);
  const makeKey = (): number => nextKey.current++;
  const [rows, setRows] = useState<Row[]>(() =>
    props.rules.map((rule) => toRow(rule, makeKey())),
  );
  const [error, setError] = useState<FieldError | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  function patch(index: number, patchFields: Partial<CorrectionRuleProps>): void {
    setRows((current) =>
      current.map((row, i) => (i === index ? { ...row, ...patchFields } : row)),
    );
    setSaved(false);
  }

  function addRow(): void {
    setRows((current) => [
      ...current,
      toRow({ id: "", match: "", replace: "", case_sensitive: true, whole_word: true }, makeKey()),
    ]);
    setSaved(false);
  }

  function removeRow(index: number): void {
    setRows((current) => current.filter((_, i) => i !== index));
    setError(null);
    setSaved(false);
  }

  function moveRow(index: number, delta: number): void {
    setRows((current) => {
      const target = index + delta;
      if (target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
    setError(null);
    setSaved(false);
  }

  // Cheap client pre-check: non-empty fields, length bounds, rule count. The
  // subtle boundary-aware idempotence + duplicate-id checks are left to the
  // server so the client can never disagree with the authoritative gate.
  function preCheck(): FieldError | null {
    if (rows.length > limits.maxRules) {
      return { message: `You can have at most ${limits.maxRules} correction rules.`, row: null };
    }
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (!row.match.trim() || !row.replace.trim()) {
        return { message: `Rule ${i + 1}: both “find” and “replace with” are required.`, row: i };
      }
      if (row.match.length > limits.maxMatchChars) {
        return {
          message: `Rule ${i + 1}: “find” is too long (max ${limits.maxMatchChars} characters).`,
          row: i,
        };
      }
      if (row.replace.length > limits.maxReplacementChars) {
        return {
          message: `Rule ${i + 1}: “replace with” is too long (max ${limits.maxReplacementChars} characters).`,
          row: i,
        };
      }
    }
    return null;
  }

  async function save(): Promise<void> {
    const clientError = preCheck();
    if (clientError) {
      setError(clientError);
      return;
    }
    setError(null);
    setSaving(true);
    setSaved(false);
    const payload = rows.map((row) => ({
      id: row.id,
      match: row.match,
      replace: row.replace,
      case_sensitive: row.case_sensitive,
      whole_word: row.whole_word,
    }));
    try {
      const res = await fetch(action, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body: new URLSearchParams({ rules: JSON.stringify(payload), csrf_token: csrfToken }),
      });
      const body = (await res.json()) as SaveEnvelope;
      if (res.ok && body.ok && body.corrections) {
        // Re-hydrate from the server's canonical result so generated ids stick.
        setRows(body.corrections.map((rule) => toRow(rule, makeKey())));
        setSaved(true);
        return;
      }
      setError({
        message: body.error ?? "The corrections could not be saved.",
        row: body.row ?? null,
      });
    } catch {
      setError({ message: "The corrections could not be saved (network error).", row: null });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="corrections-editor my-2">
      {error && error.row === null && (
        <p className="notice" role="alert">
          {error.message}
        </p>
      )}
      {rows.length === 0 && <p className="muted">No correction rules yet.</p>}
      <ol className="corrections-rows">
        {rows.map((row, index) => {
          const rowError = error && error.row === index ? error.message : null;
          return (
            <li key={row.key} className="corrections-row field my-2">
              <div className="flex items-center my-1">
                <label className="text-sm mr-2">
                  Find
                  <input
                    className="w-full text-sm"
                    type="text"
                    value={row.match}
                    aria-label={`Rule ${index + 1} find`}
                    maxLength={limits.maxMatchChars}
                    onChange={(e) => patch(index, { match: e.target.value })}
                  />
                </label>
                <label className="text-sm mr-2">
                  Replace with
                  <input
                    className="w-full text-sm"
                    type="text"
                    value={row.replace}
                    aria-label={`Rule ${index + 1} replace with`}
                    maxLength={limits.maxReplacementChars}
                    onChange={(e) => patch(index, { replace: e.target.value })}
                  />
                </label>
              </div>
              <div className="flex items-center my-1 text-sm">
                <label className="mr-2">
                  <input
                    type="checkbox"
                    className="mr-2"
                    checked={row.case_sensitive}
                    onChange={(e) => patch(index, { case_sensitive: e.target.checked })}
                  />
                  Match case
                </label>
                <label className="mr-2">
                  <input
                    type="checkbox"
                    className="mr-2"
                    checked={row.whole_word}
                    onChange={(e) => patch(index, { whole_word: e.target.checked })}
                  />
                  Whole word only
                </label>
                <button
                  type="button"
                  className="text-sm mr-2"
                  onClick={() => moveRow(index, -1)}
                  disabled={index === 0}
                  aria-label={`Move rule ${index + 1} up`}
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="text-sm mr-2"
                  onClick={() => moveRow(index, 1)}
                  disabled={index === rows.length - 1}
                  aria-label={`Move rule ${index + 1} down`}
                >
                  ↓
                </button>
                <button
                  type="button"
                  className="text-sm"
                  onClick={() => removeRow(index)}
                  aria-label={`Remove rule ${index + 1}`}
                >
                  Remove
                </button>
              </div>
              {rowError && (
                <p className="notice" role="alert">
                  {rowError}
                </p>
              )}
            </li>
          );
        })}
      </ol>
      <div className="flex items-center my-2">
        <button
          type="button"
          className="mr-2"
          onClick={addRow}
          disabled={rows.length >= limits.maxRules}
        >
          Add rule
        </button>
        <button type="button" onClick={() => void save()} disabled={saving}>
          {saving ? "Saving…" : "Save corrections"}
        </button>
        {saved && (
          <span className="muted text-sm ml-2" role="status">
            Saved.
          </span>
        )}
      </div>
    </div>
  );
}
