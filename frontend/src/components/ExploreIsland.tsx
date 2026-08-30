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

export function ExploreIsland({
  rows,
  total,
  query,
  page,
  pageSize,
  stats,
  termStats,
}: ExploreIslandProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const terms = termStats ?? [];
  const statTiles = [
    [integerFormatter.format(stats.total_segments), "segments"],
    [integerFormatter.format(stats.total_runs), "recordings"],
    [integerFormatter.format(stats.total_speakers), "speakers"],
    [Math.round(stats.total_hours).toLocaleString(), "hours"],
  ];

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

      {!query ? (
        stats.total_runs === 0 ? (
          <div className="py-12 text-center text-base text-[var(--ink-3)]">
            No transcripts yet. Submit a recording to start exploring.
          </div>
        ) : terms.length === 0 ? (
          <div className="py-12 text-center text-base text-[var(--ink-3)]">
            Type a term to search across all transcripts
          </div>
        ) : null
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
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const paletteIndex = speakerPaletteIndex(row.speaker_id);
                  return (
                    <tr
                      className="border-b border-[var(--line)] hover:bg-[var(--surface-2)]"
                      key={`${row.segment_id}-${row.start_seconds}`}
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
                    </tr>
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
