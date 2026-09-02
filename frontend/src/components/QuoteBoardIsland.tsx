import { useState, useCallback, useRef } from "react";

export interface SavedQuoteRow {
  id: string;
  search_query: string;
  left_context: string;
  hit: string;
  right_context: string;
  speaker_name: string | null;
  media_title: string;
  run_id: string;
  start_seconds: number;
  note: string | null;
  created_at: string | null;
}

export interface QuoteBoardProps {
  quotes: SavedQuoteRow[];
  total: number;
  projectId: string;
  projectName: string;
  csrfToken: string;
}

function NoteCell({
  quote,
  csrfToken,
}: {
  quote: SavedQuoteRow;
  csrfToken: string;
}) {
  const [lastSaved, setLastSaved] = useState(quote.note ?? "");
  const [note, setNote] = useState(quote.note ?? "");
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState(false);
  const savingRef = useRef(false);

  const save = useCallback(async () => {
    if (savingRef.current || note === lastSaved) {
      setEditing(false);
      return;
    }
    savingRef.current = true;
    setError(false);
    const body = new FormData();
    body.append("note", note);
    body.append("csrf_token", csrfToken);
    try {
      const res = await fetch(`/quotes/${quote.id}`, {
        method: "PATCH",
        body,
      });
      if (res.ok) {
        const data = await res.json();
        const saved = data.note ?? "";
        setLastSaved(saved);
        setNote(saved);
        setEditing(false);
      } else {
        setError(true);
      }
    } catch {
      setError(true);
    } finally {
      savingRef.current = false;
    }
  }, [note, lastSaved, csrfToken, quote.id]);

  if (editing) {
    return (
      <div>
        <input
          type="text"
          className="w-full rounded-[var(--r-sm)] border border-[var(--line)] bg-[var(--surface)] px-1 py-0.5 text-sm text-[var(--ink)]"
          style={error ? { borderColor: "var(--warn)" } : undefined}
          value={note}
          onChange={(e) => { setNote(e.target.value); setError(false); }}
          onBlur={save}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              (e.target as HTMLInputElement).blur();
            }
            if (e.key === "Escape") {
              setNote(lastSaved);
              setEditing(false);
            }
          }}
          disabled={savingRef.current}
          maxLength={2000}
          aria-label="Edit note"
          autoFocus
        />
        {error ? (
          <span className="text-[length:var(--t-micro)] text-[var(--warn)]">
            Save failed
          </span>
        ) : null}
      </div>
    );
  }

  return (
    <button
      type="button"
      className="w-full text-left text-sm"
      style={{
        background: "transparent",
        border: "none",
        padding: "2px 4px",
        cursor: "pointer",
        color: lastSaved ? "var(--ink-2)" : "var(--ink-3)",
        minHeight: "1.5em",
      }}
      onClick={() => setEditing(true)}
      title="Click to edit note"
    >
      {lastSaved || "add note…"}
    </button>
  );
}

export function QuoteBoardIsland({
  quotes: initialQuotes,
  total: initialTotal,
  projectId,
  csrfToken,
}: QuoteBoardProps) {
  const [quotes, setQuotes] = useState(initialQuotes);
  const [total, setTotal] = useState(initialTotal);

  const handleDelete = useCallback(
    async (quoteId: string) => {
      const body = new FormData();
      body.append("csrf_token", csrfToken);
      const res = await fetch(`/quotes/${quoteId}`, {
        method: "DELETE",
        body,
      });
      if (res.ok) {
        setQuotes((prev) => prev.filter((q) => q.id !== quoteId));
        setTotal((prev) => prev - 1);
      }
    },
    [csrfToken],
  );

  if (quotes.length === 0) {
    return (
      <p
        className="text-sm text-[var(--ink-3)]"
        style={{ padding: "var(--space-3) 0" }}
      >
        Save quotes from Explore to build your evidence board.
      </p>
    );
  }

  return (
    <>
      <div className="max-w-full overflow-x-auto">
        <table
          className="w-full border-collapse text-sm"
          role="grid"
          aria-label="Saved quotes"
        >
          <thead>
            <tr className="border-b-2 border-[var(--line)] text-[length:var(--t-micro)] uppercase tracking-wide text-[var(--ink-3)]">
              <th className="px-2 py-1 text-left font-semibold" scope="col">
                Query
              </th>
              <th className="px-2 py-1 text-right font-semibold" scope="col">
                Context
              </th>
              <th className="px-2 py-1 text-left font-semibold" scope="col">
                Match
              </th>
              <th className="px-2 py-1 text-left font-semibold" scope="col">
                Context
              </th>
              <th className="px-2 py-1 text-left font-semibold" scope="col">
                Source
              </th>
              <th
                className="px-2 py-1 text-left font-semibold"
                scope="col"
                style={{ minWidth: "120px" }}
              >
                Note
              </th>
              <th className="w-8 px-1 py-1" scope="col">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {quotes.map((q) => (
              <tr
                className="border-b border-[var(--line)] hover:bg-[var(--surface-2)]"
                key={q.id}
              >
                <td className="max-w-[12ch] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-[var(--ink-2)]">
                  <a
                    className="text-[var(--accent)] no-underline hover:underline"
                    href={`/explore?q=${encodeURIComponent(q.search_query)}&project=${projectId}`}
                    title={q.search_query}
                  >
                    {q.search_query}
                  </a>
                </td>
                <td className="max-w-[18ch] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-right text-[var(--ink-2)] opacity-70">
                  {q.left_context}
                </td>
                <td className="whitespace-nowrap px-2 py-1 font-semibold">
                  <mark className="rounded-sm bg-[var(--accent-soft)] px-0.5 text-[var(--accent)]">
                    {q.hit}
                  </mark>
                </td>
                <td className="max-w-[18ch] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-[var(--ink-2)] opacity-70">
                  {q.right_context}
                </td>
                <td className="whitespace-nowrap px-2 py-1 text-[length:var(--t-micro)]">
                  {q.speaker_name ? (
                    <span className="mr-1 inline-block rounded-[var(--r-sm)] border border-[var(--line)] px-1 text-[var(--ink-2)]">
                      {q.speaker_name}
                    </span>
                  ) : null}
                  <a
                    className="text-[var(--accent)] no-underline hover:underline"
                    href={`/runs/${q.run_id}/transcript?t=${q.start_seconds}`}
                  >
                    {q.media_title} · {q.start_seconds.toFixed(1)}s
                  </a>
                </td>
                <td className="px-1 py-1">
                  <NoteCell quote={q} csrfToken={csrfToken} />
                </td>
                <td className="px-1 py-1 text-center">
                  <button
                    type="button"
                    className="inline-flex h-6 w-6 items-center justify-center rounded-[var(--r-sm)] text-sm text-[var(--ink-3)] hover:bg-[var(--warn-soft)] hover:text-[var(--warn)]"
                    style={{ background: "transparent", border: "none", cursor: "pointer" }}
                    onClick={() => handleDelete(q.id)}
                    title="Remove from quote board"
                    aria-label={`Remove quote: ${q.hit}`}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {total > quotes.length ? (
        <p className="mt-2 text-sm text-[var(--ink-3)]">
          Showing {quotes.length} of {total} quotes.
        </p>
      ) : null}
    </>
  );
}
