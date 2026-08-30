import { useState } from "react";

import {
  TemporalTrendChart,
  type TemporalBucket,
  type TemporalChartSeries,
} from "./TemporalTrendChart";

interface TemporalSeries {
  key: string;
  label: string;
  total_count: number;
  recording_count: number;
  values: number[];
}

interface EntityTemporalSeries extends TemporalSeries {
  kind: string | null;
}

export interface TemporalTrendsProps {
  schema_version: number;
  algorithm_version: string;
  range: {
    start: string | null;
    end: string | null;
    bucket_unit: string | null;
    week_starts_on: string;
    timezone: string;
  };
  buckets: TemporalBucket[];
  terms: TemporalSeries[];
  entities: EntityTemporalSeries[];
  date_provenance: {
    preference: string[];
    source_upload_date_recordings: number;
    ingestion_created_at_recordings: number;
    undated_recordings: number;
    label: string;
  };
  coverage: {
    dated_recordings: number;
    term_recordings: number;
    entity_enriched_recordings: number;
  };
  truncated: {
    terms: boolean;
    entities: boolean;
  };
}

type Mode = "terms" | "entities";

const MAX_VISIBLE = 5;
const SERIES_COLORS = [
  "var(--spk-0)",
  "var(--spk-1)",
  "var(--spk-2)",
  "var(--spk-3)",
  "var(--spk-4)",
];

function formatRange(start: string | null, end: string | null): string {
  if (!start || !end) return "No recording date range";
  const formatter = new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeZone: "UTC",
  });
  return `${formatter.format(new Date(`${start}T00:00:00Z`))} – ${formatter.format(new Date(`${end}T00:00:00Z`))}`;
}

export function TemporalTrendsIsland(props: TemporalTrendsProps) {
  const [mode, setMode] = useState<Mode>("terms");
  const [termKeys, setTermKeys] = useState(
    () => new Set(props.terms.slice(0, MAX_VISIBLE).map((item) => item.key)),
  );
  const [entityKeys, setEntityKeys] = useState(
    () =>
      new Set(props.entities.slice(0, MAX_VISIBLE).map((item) => item.key)),
  );
  const options = mode === "terms" ? props.terms : props.entities;
  const selectedKeys = mode === "terms" ? termKeys : entityKeys;
  const selectedList = options.filter((item) => selectedKeys.has(item.key));
  const colorByKey = new Map(
    selectedList.map((item, index) => [
      item.key,
      SERIES_COLORS[index % SERIES_COLORS.length],
    ]),
  );
  const selectedSeries: TemporalChartSeries[] = selectedList.map((item) => ({
    ...item,
    color: colorByKey.get(item.key) ?? SERIES_COLORS[0],
  }));

  const toggleSeries = (key: string) => {
    const setter = mode === "terms" ? setTermKeys : setEntityKeys;
    setter((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else if (next.size < MAX_VISIBLE) next.add(key);
      return next;
    });
  };

  const noDatedRecordings =
    props.coverage.dated_recordings === 0 || !props.buckets.length;
  const noModeData = options.length === 0;
  const unit = props.range.bucket_unit ?? "calendar";

  return (
    <section aria-labelledby="temporal-trends-heading">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3
            id="temporal-trends-heading"
            className="m-0 text-[length:var(--t-xs)] font-semibold tracking-[0.06em] text-[var(--ink-2)]"
          >
            TEMPORAL TRENDS
          </h3>
          <p className="m-0 mt-1 text-[length:var(--t-xs)] text-[var(--ink-3)]">
            {formatRange(props.range.start, props.range.end)} · {unit} buckets
          </p>
        </div>
        <div className="flex gap-1" role="group" aria-label="Trend type">
          {(["terms", "entities"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={item === mode ? "primary small" : "secondary small"}
              aria-pressed={item === mode}
              onClick={() => setMode(item)}
            >
              {item === "terms" ? "Terms" : "Entities"}
            </button>
          ))}
        </div>
      </div>

      {noDatedRecordings ? (
        <p className="py-8 text-center text-sm text-[var(--ink-3)]">
          No dated recordings are available yet.
        </p>
      ) : noModeData ? (
        <p className="py-8 text-center text-sm text-[var(--ink-3)]">
          {mode === "terms"
            ? "No transcript terms are available for these recordings."
            : "No entity enrichment is available for these recordings."}
        </p>
      ) : (
        <>
          <fieldset className="mb-2 flex flex-wrap gap-x-3 gap-y-1 border-0 p-0">
            <legend className="sr-only">Visible {mode}</legend>
            {options.map((item) => (
              <label
                key={item.key}
                className="inline-flex cursor-pointer items-center gap-1 text-[length:var(--t-xs)] text-[var(--ink-2)]"
              >
                <input
                  type="checkbox"
                  checked={selectedKeys.has(item.key)}
                  onChange={() => toggleSeries(item.key)}
                />
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{
                    backgroundColor:
                      colorByKey.get(item.key) ?? "var(--line)",
                  }}
                  aria-hidden="true"
                />
                {item.label}
                <span className="text-[var(--ink-3)]">
                  ({item.total_count})
                </span>
              </label>
            ))}
          </fieldset>

          {selectedSeries.length ? (
            <TemporalTrendChart
              buckets={props.buckets}
              series={selectedSeries}
              bucketUnit={props.range.bucket_unit}
            />
          ) : (
            <p className="py-8 text-center text-sm text-[var(--ink-3)]">
              Select a series to display it on the chart.
            </p>
          )}

          <div
            className="mt-1 flex flex-wrap gap-x-3 gap-y-1"
            aria-label="Trend legend"
          >
            {selectedSeries.map((item) => (
              <span
                key={item.key}
                className="inline-flex items-center gap-1 text-[length:var(--t-xs)] text-[var(--ink-2)]"
              >
                <span
                  className="inline-block h-0.5 w-4"
                  style={{ backgroundColor: item.color }}
                  aria-hidden="true"
                />
                {item.label}
              </span>
            ))}
          </div>
        </>
      )}

      <p className="mb-0 mt-3 text-[length:var(--t-xs)] text-[var(--ink-3)]">
        {props.date_provenance.label} Source dates:{" "}
        {props.date_provenance.source_upload_date_recordings}; ingestion
        fallback: {props.date_provenance.ingestion_created_at_recordings}.
      </p>
    </section>
  );
}
