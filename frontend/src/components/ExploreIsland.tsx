import { Fragment, useState, useCallback, useEffect, useRef } from "react";
import { MeaningMap } from "./MeaningMap";
import { TermBarChart } from "./TermBarChart";
import { WordCloud, type TermDatum } from "./WordCloud";

export interface KWICRow {
  left_context: string;
  hit: string;
  right_context: string;
  speaker_name: string | null;
  speaker_id: string | null;
  run_id: string;
  media_title: string;
  segment_id: string;
  start_seconds: number;
  confidence: number | null;
  suspect: boolean;
}

export interface ExploreIslandProps {
  rows: KWICRow[];
  total: number;
  query: string;
  page: number;
  pageSize: number;
  stats: {
    total_segments: number;
    total_runs: number;
    total_speakers: number;
    total_hours: number;
  };
  filters: {
    project_id: string | null;
    speaker_id: string | null;
    date_from: string | null;
    date_to: string | null;
    low_confidence_only: boolean;
    suspect_only: boolean;
  };
  termStats?: TermDatum[];
  tagStats?: TagStat[];
  csrfQuoteSave?: string;
}

export interface TagStat {
  tag_id: string;
  name: string;
  color: number;
  count: number;
}

function searchHref(term: string): string {
  const params = new URLSearchParams(window.location.search);
  params.set("q", term);
  params.delete("page");
  return `/explore?${params.toString()}`;
}

function pageHref(page: number): string {
  const params = new URLSearchParams(window.location.search);
  params.set("page", String(page));
  return `/explore?${params.toString()}`;
}

function speakerPaletteIndex(id: string | null): number {
  if (!id) return 0;
  let hash = 0;
  for (const character of id) hash = (hash * 31 + character.charCodeAt(0)) | 0;
  return Math.abs(hash) % 8;
}

const integerFormatter = new Intl.NumberFormat();

function handleTermClick(term: string) {
  window.location.href = searchHref(term);
}

interface SimilarItem {
  run_id: string;
  title: string | null;
  speaker_label: string | null;
  start_seconds: number;
  end_seconds: number;
  preview: string;
  jump_url: string;
}

interface SimilarResponse {
  state: string;
  items: SimilarItem[];
}

const SIMILAR_STATE_COPY: Record<string, string> = {
  off: "The semantic index is disabled, so similar passages are unavailable.",
  unavailable: "The semantic model is not installed, so similar passages are unavailable.",
  indexing: "Passages are still being indexed. Try again once processing finishes.",
  not_found: "This row's segment no longer exists.",
  empty_text: "This row has no text to match against.",
};

/** The expanded "More like this" detail row (#357): lazy fetch on first
 * expand, page-lifetime cache keyed by segment, honest state copy, and an
 * abort on collapse so a slow response cannot land on a different row. */
function SimilarPanel({
  segmentId,
  colSpan,
  cache,
  panelId,
}: {
  segmentId: string;
  colSpan: number;
  cache: Map<string, SimilarResponse>;
  panelId: string;
}) {
  const [response, setResponse] = useState<SimilarResponse | null>(
    () => cache.get(segmentId) ?? null,
  );
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (cache.has(segmentId)) return;
    const controller = new AbortController();
    fetch(`/explore/segments/${segmentId}/similar`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.json() as Promise<SimilarResponse>;
      })
      .then((body) => {
        // Only settled answers are cached: "indexing" (and friends) are
        // transient, and pinning one for the page lifetime would keep saying
        // "still indexing" after indexing finished.
        if (body.state === "ok") cache.set(segmentId, body);
        setResponse(body);
      })
      .catch((error: unknown) => {
        if ((error as { name?: string }).name !== "AbortError") setFailed(true);
      });
    return () => controller.abort();
  }, [segmentId, cache]);

  let content;
  if (failed) {
    content = (
      <p className="text-sm text-[var(--ink-3)]">
        Similar passages could not load. Try again after a refresh.
      </p>
    );
  } else if (!response) {
    content = <p className="text-sm text-[var(--ink-3)]">Finding similar passages…</p>;
  } else if (response.state !== "ok") {
    content = (
      <p className="text-sm text-[var(--ink-3)]">
        {SIMILAR_STATE_COPY[response.state] ?? "Similar passages are unavailable."}
      </p>
    );
  } else if (response.items.length === 0) {
    content = (
      <p className="text-sm text-[var(--ink-3)]">
        No other passages read like this one.
      </p>
    );
  } else {
    content = (
      <ul className="m-0 list-none space-y-2 p-0">
        {response.items.map((item) => (
          <li
            key={`${item.run_id}-${item.start_seconds}-${item.end_seconds}`}
            className="text-sm"
          >
            <span className="block text-[length:var(--t-micro)] text-[var(--ink-3)]">
              {item.speaker_label ? `${item.speaker_label} · ` : ""}
              <a
                className="text-[var(--accent)] no-underline hover:underline"
                href={item.jump_url}
              >
                {item.title ?? "Untitled recording"} ·{" "}
                {item.start_seconds.toFixed(1)}s
              </a>
            </span>
            <span className="block text-[var(--ink-2)]">{item.preview}</span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <tr id={panelId} className="border-b border-[var(--line)] bg-[var(--surface-2)]">
      <td className="px-3 py-2" colSpan={colSpan}>
        <p className="mb-1 mt-0 text-[length:var(--t-micro)] uppercase tracking-wide text-[var(--ink-3)]">
          More like this passage
        </p>
        {content}
      </td>
    </tr>
  );
}

type SaveState = "idle" | "saving" | "saved" | "duplicate" | "no-project" | "error";

function SaveButton({
  row,
  query,
  csrfToken,
  savedSet,
  onSaved,
}: {
  row: KWICRow;
  query: string;
  csrfToken: string;
  savedSet: Set<string>;
  onSaved: (key: string) => void;
}) {
  const key = `${row.segment_id}:${query}`;
  const alreadySaved = savedSet.has(key);
  const [state, setState] = useState<SaveState>(alreadySaved ? "saved" : "idle");

  const handleClick = useCallback(async () => {
    if (state === "saved" || state === "saving") return;
    setState("saving");
    const body = new FormData();
    body.append("segment_id", row.segment_id);
    body.append("run_id", row.run_id);
    body.append("search_query", query);
    body.append("left_context", row.left_context);
    body.append("hit", row.hit);
    body.append("right_context", row.right_context);
    body.append("media_title", row.media_title);
    body.append("start_seconds", String(row.start_seconds));
    body.append("csrf_token", csrfToken);
    if (row.speaker_name) body.append("speaker_name", row.speaker_name);

    try {
      const res = await fetch("/quotes", { method: "POST", body });
      if (res.status === 201) {
        setState("saved");
        onSaved(key);
      } else if (res.status === 409) {
        setState("duplicate");
        onSaved(key);
      } else if (res.status === 422) {
        setState("no-project");
      } else {
        setState("error");
      }
    } catch {
      setState("error");
    }
  }, [row, query, csrfToken, state, key, onSaved]);

  const label =
    state === "saved" || state === "duplicate"
      ? "✓"
      : state === "saving"
        ? "…"
        : state === "no-project"
          ? "—"
          : state === "error"
            ? "!"
            : "⭳";

  const title =
    state === "saved"
      ? "Saved to project"
      : state === "duplicate"
        ? "Already saved"
        : state === "saving"
          ? "Saving…"
          : state === "no-project"
            ? "Recording not in a project"
            : state === "error"
              ? "Save failed"
              : "Save to quote board";

  return (
    <button
      type="button"
      className="inline-flex h-6 w-6 items-center justify-center rounded-[var(--r-sm)] text-sm hover:bg-[var(--surface-2)]"
      style={{
        opacity: state === "saved" || state === "duplicate" ? 0.5 : 1,
        cursor:
          state === "saved" || state === "saving" || state === "duplicate"
            ? "default"
            : "pointer",
        border: "none",
        background: "transparent",
        color: "var(--ink-2)",
      }}
      onClick={handleClick}
      title={title}
      aria-label={title}
      disabled={state === "saving"}
    >
      {label}
    </button>
  );
}

export function ExploreIsland({
  rows,
  total,
  query,
  page,
  pageSize,
  stats,
  filters,
  termStats,
  tagStats,
  csrfQuoteSave,
}: ExploreIslandProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const terms = termStats ?? [];
  const tags = tagStats ?? [];
  const statTiles = [
    [integerFormatter.format(stats.total_segments), "segments"],
    [integerFormatter.format(stats.total_runs), "recordings"],
    [integerFormatter.format(stats.total_speakers), "speakers"],
    [Math.round(stats.total_hours).toLocaleString(), "hours"],
  ];
  const [savedSet, setSavedSet] = useState<Set<string>>(() => new Set());
  const handleSaved = useCallback((key: string) => {
    setSavedSet((prev) => new Set(prev).add(key));
  }, []);
  const showSave = !!csrfQuoteSave && !!query;
  // KWIC emits one row per segment (one ts_headline per matching segment),
  // so the row key is unique per render; the fetch cache is per segment and
  // lives for the page.
  const [expandedSimilar, setExpandedSimilar] = useState<string | null>(null);
  const similarCache = useRef(new Map<string, SimilarResponse>()).current;
  const resultColumns = 5 + (showSave ? 1 : 0);

  return (
    <section aria-label="Explore results">
      <div
        className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4"
        aria-label="Corpus statistics"
      >
        {statTiles.map(([value, label]) => (
          <div
            className="rounded-[var(--r-md)] border border-[var(--line)] bg-[var(--surface)] px-3 py-2 text-center"
            key={label}
          >
            <span className="block font-mono text-base font-semibold text-[var(--ink)]">
              {value}
            </span>
            <span className="text-[length:var(--t-micro)] uppercase tracking-wide text-[var(--ink-3)]">
              {label}
            </span>
          </div>
        ))}
      </div>

      {terms.length > 0 ? (
        <div className="explore-term-panel mb-4">
          <h2 className="explore-term-heading">Top terms</h2>
          <div className="explore-term-grid">
            <div className="explore-term-cloud">
              <WordCloud terms={terms} onTermClick={handleTermClick} />
            </div>
            <div className="explore-term-bars">
              <TermBarChart terms={terms} onTermClick={handleTermClick} />
            </div>
          </div>
        </div>
      ) : null}

      <div className="explore-term-panel mb-4">
        <h2 className="explore-term-heading">Highlight tags</h2>
        {tags.length > 0 ? (
          <TermBarChart
            terms={tags.map((t) => ({
              term: t.name,
              count: t.count,
              doc_count: 0,
              tfidf: 0,
            }))}
            ariaLabel="Highlight tags by annotation count"
          />
        ) : (
          <p className="text-sm text-[var(--ink-3)]">No highlight tags yet.</p>
        )}
      </div>

      <MeaningMap projectId={filters.project_id} />

      {!query ? (
        stats.total_runs === 0 ? (
          <div className="py-12 text-center text-base text-[var(--ink-3)]">
            No transcripts yet. Submit a recording to start exploring.
          </div>
        ) : (
          <div className="explore-empty-state">
            <p>
              Search finds every occurrence across all your transcripts, with
              context, speakers, and timestamps.
            </p>
            {terms.length > 0 && (
              <p style={{ marginTop: "var(--space-2)" }}>
                Try:{" "}
                {terms.slice(0, 3).map((t, i) => {
                  const params = new URLSearchParams({ q: t.term });
                  if (filters.project_id) params.set("project", filters.project_id);
                  if (filters.speaker_id) params.set("speaker", filters.speaker_id);
                  return (
                    <span key={t.term}>
                      {i > 0 && " · "}
                      <a href={`/explore?${params}`}>{t.term}</a>
                    </span>
                  );
                })}
              </p>
            )}
            <p
              className="oc-muted"
              style={{
                marginTop: "var(--space-2)",
                fontSize: "var(--t-sm)",
              }}
            >
              After searching, you will see a word cloud, a meaning map, and
              concordance matches you can export.
            </p>
          </div>
        )
      ) : total === 0 ? (
        <div className="py-12 text-center text-base text-[var(--ink-3)]">
          No matches for &apos;{query}&apos;
        </div>
      ) : (
        <>
          <div className="max-w-full overflow-x-auto">
            <table
              className="w-full border-collapse text-sm"
              role="grid"
              aria-label="Concordance matches"
            >
              <thead>
                <tr className="border-b-2 border-[var(--line)] text-[length:var(--t-micro)] uppercase tracking-wide text-[var(--ink-3)]">
                  <th
                    className="px-2 py-1 text-right font-semibold"
                    scope="col"
                  >
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
                  <th className="w-8 px-1 py-1" scope="col">
                    <span className="sr-only">More like this</span>
                  </th>
                  {showSave ? (
                    <th className="w-8 px-1 py-1" scope="col">
                      <span className="sr-only">Save</span>
                    </th>
                  ) : null}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const paletteIndex = speakerPaletteIndex(row.speaker_id);
                  const rowKey = `${row.segment_id}-${row.start_seconds}`;
                  const isExpanded = expandedSimilar === rowKey;
                  return (
                    <Fragment key={rowKey}>
                    <tr
                      className="border-b border-[var(--line)] hover:bg-[var(--surface-2)]"
                    >
                      <td className="max-w-[25ch] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-right text-[var(--ink-2)] opacity-70">
                        {row.left_context}
                      </td>
                      <td className="whitespace-nowrap px-2 py-1 font-semibold">
                        <mark className="rounded-sm bg-[var(--accent-soft)] px-0.5 text-[var(--accent)]">
                          {row.hit}
                        </mark>
                      </td>
                      <td className="max-w-[25ch] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-[var(--ink-2)] opacity-70">
                        {row.right_context}
                      </td>
                      <td className="whitespace-nowrap px-2 py-1 text-[length:var(--t-micro)]">
                        {row.speaker_name ? (
                          <span
                            className={`spk-${paletteIndex} mr-1 inline-block rounded-[var(--r-sm)] border border-[var(--spk-accent)] px-1 text-[var(--spk-accent)]`}
                          >
                            {row.speaker_name}
                          </span>
                        ) : null}
                        <a
                          className="text-[var(--accent)] no-underline hover:underline"
                          href={`/runs/${row.run_id}/transcript#t=${row.start_seconds}`}
                        >
                          {row.media_title} · {row.start_seconds.toFixed(1)}s
                        </a>
                        {row.suspect ? (
                          <span className="ml-1 inline-block rounded-[var(--r-sm)] bg-[var(--warn-soft)] px-1 text-[var(--warn)]">
                            suspect
                          </span>
                        ) : null}
                        {row.confidence !== null && row.confidence < 0.7 ? (
                          <span className="ml-1 inline-block rounded-[var(--r-sm)] bg-[var(--surface-2)] px-1 text-[var(--ink-3)]">
                            low confidence
                          </span>
                        ) : null}
                      </td>
                      <td className="px-1 py-1 text-center">
                        <button
                          type="button"
                          className="inline-flex h-6 w-6 items-center justify-center rounded-[var(--r-sm)] text-sm hover:bg-[var(--surface-2)]"
                          style={{
                            border: "none",
                            background: "transparent",
                            color: isExpanded ? "var(--accent)" : "var(--ink-2)",
                            cursor: "pointer",
                          }}
                          onClick={() =>
                            setExpandedSimilar(isExpanded ? null : rowKey)
                          }
                          title="More like this passage"
                          aria-label="More like this passage"
                          aria-expanded={isExpanded}
                          aria-controls={`similar-${row.segment_id}`}
                        >
                          ≈
                        </button>
                      </td>
                      {showSave ? (
                        <td className="px-1 py-1 text-center">
                          <SaveButton
                            row={row}
                            query={query}
                            csrfToken={csrfQuoteSave!}
                            savedSet={savedSet}
                            onSaved={handleSaved}
                          />
                        </td>
                      ) : null}
                    </tr>
                    {isExpanded ? (
                      <SimilarPanel
                        segmentId={row.segment_id}
                        colSpan={resultColumns}
                        cache={similarCache}
                        panelId={`similar-${row.segment_id}`}
                      />
                    ) : null}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          {pageCount > 1 ? (
            <nav
              className="mt-4 flex items-center justify-center gap-3"
              aria-label="Results pages"
            >
              {page > 1 ? (
                <a className="secondary" href={pageHref(page - 1)}>
                  ← Previous
                </a>
              ) : null}
              <span className="text-sm text-[var(--ink-3)]">
                Page {page} of {pageCount}
              </span>
              {page < pageCount ? (
                <a className="secondary" href={pageHref(page + 1)}>
                  Next →
                </a>
              ) : null}
            </nav>
          ) : null}
        </>
      )}
    </section>
  );
}
