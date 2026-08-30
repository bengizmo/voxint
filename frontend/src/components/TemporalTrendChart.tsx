import { extent, max } from "d3-array";
import { scaleLinear, scaleUtc } from "d3-scale";
import { line } from "d3-shape";
import { utcFormat } from "d3-time-format";
import { useEffect, useMemo, useRef, useState } from "react";

export interface TemporalBucket {
  start: string;
  end_exclusive: string;
  recording_count: number;
  date_sources: {
    source_upload_date: number;
    ingestion_created_at: number;
  };
}

export interface TemporalChartSeries {
  key: string;
  label: string;
  values: number[];
  color: string;
}

interface TemporalTrendChartProps {
  buckets: TemporalBucket[];
  series: TemporalChartSeries[];
  bucketUnit: string | null;
}

interface ChartPoint {
  bucket: TemporalBucket;
  bucketIndex: number;
  date: Date;
  value: number;
  series: TemporalChartSeries;
}

const HEIGHT = 300;
const MARGIN = { top: 16, right: 18, bottom: 42, left: 46 };
const FALLBACK_WIDTH = 720;

const formatDay = utcFormat("%b %-d, %Y");
const formatMonth = utcFormat("%b %Y");
const formatYear = utcFormat("%Y");

function parseDate(value: string): Date {
  return new Date(`${value}T00:00:00Z`);
}

function bucketMidpoint(bucket: TemporalBucket): Date {
  const start = parseDate(bucket.start).getTime();
  const end = parseDate(bucket.end_exclusive).getTime();
  return new Date(start + (end - start) / 2);
}

function bucketLabel(bucket: TemporalBucket): string {
  const start = parseDate(bucket.start);
  const end = parseDate(bucket.end_exclusive);
  if (bucket.end_exclusive === bucket.start) return formatDay(start);
  return `${formatDay(start)} – ${formatDay(end)} (exclusive)`;
}

function tickFormatter(bucketUnit: string | null, rangeInDays: number) {
  if (rangeInDays > 1_500) return formatYear;
  if (bucketUnit === "month" || rangeInDays > 400) return formatMonth;
  return utcFormat("%b %-d");
}

export function TemporalTrendChart({
  buckets,
  series,
  bucketUnit,
}: TemporalTrendChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FALLBACK_WIDTH);
  const [hovered, setHovered] = useState<ChartPoint | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateWidth = (nextWidth: number) => {
      if (nextWidth > 0) setWidth(Math.max(320, Math.round(nextWidth)));
    };
    updateWidth(container.getBoundingClientRect().width);

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) updateWidth(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const model = useMemo(() => {
    const plotWidth = Math.max(1, width - MARGIN.left - MARGIN.right);
    const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
    const start = parseDate(buckets[0]?.start ?? "1970-01-01");
    const end = parseDate(buckets.at(-1)?.end_exclusive ?? "1970-01-02");
    const x = scaleUtc()
      .domain([start, end])
      .range([MARGIN.left, MARGIN.left + plotWidth]);
    const largest = max(series.flatMap((item) => item.values)) ?? 0;
    const y = scaleLinear()
      .domain([0, Math.max(1, largest)])
      .nice()
      .range([MARGIN.top + plotHeight, MARGIN.top]);
    const points = series.map((item) =>
      buckets.map<ChartPoint>((bucket, bucketIndex) => ({
        bucket,
        bucketIndex,
        date: bucketMidpoint(bucket),
        value: item.values[bucketIndex] ?? 0,
        series: item,
      })),
    );
    const path = line<ChartPoint>()
      .x((point) => x(point.date))
      .y((point) => y(point.value));
    const dateExtent = extent([start, end]);
    const rangeInDays =
      ((dateExtent[1]?.getTime() ?? end.getTime()) -
        (dateExtent[0]?.getTime() ?? start.getTime())) /
      86_400_000;

    return {
      x,
      y,
      points,
      paths: points.map((item) => path(item) ?? ""),
      xTicks: x.ticks(Math.min(6, Math.max(2, buckets.length))),
      yTicks: y.ticks(5),
      formatTick: tickFormatter(bucketUnit, rangeInDays),
      baseline: MARGIN.top + plotHeight,
    };
  }, [bucketUnit, buckets, series, width]);

  const tooltipX = hovered ? model.x(hovered.date) : 0;
  const tooltipY = hovered ? model.y(hovered.value) : 0;
  const tooltipWidth = 190;
  const tooltipLeft = Math.min(
    Math.max(MARGIN.left, tooltipX - tooltipWidth / 2),
    width - MARGIN.right - tooltipWidth,
  );

  return (
    <div ref={containerRef} className="w-full overflow-hidden">
      <svg
        className="block w-full"
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label="Term and entity frequency over recording dates"
      >
        <title>Term and entity frequency over recording dates</title>

        {model.yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={MARGIN.left}
              x2={width - MARGIN.right}
              y1={model.y(tick)}
              y2={model.y(tick)}
              stroke="var(--line)"
            />
            <text
              x={MARGIN.left - 8}
              y={model.y(tick)}
              dy="0.32em"
              textAnchor="end"
              fill="var(--ink-3)"
              fontSize="10"
            >
              {tick}
            </text>
          </g>
        ))}

        <line
          x1={MARGIN.left}
          x2={width - MARGIN.right}
          y1={model.baseline}
          y2={model.baseline}
          stroke="var(--line-strong)"
        />
        {model.xTicks.map((tick) => (
          <g
            key={`x-${tick.toISOString()}`}
            transform={`translate(${model.x(tick)},0)`}
          >
            <line
              y1={model.baseline}
              y2={model.baseline + 5}
              stroke="var(--line-strong)"
            />
            <text
              y={model.baseline + 18}
              textAnchor="middle"
              fill="var(--ink-3)"
              fontSize="10"
            >
              {model.formatTick(tick)}
            </text>
          </g>
        ))}

        {series.map((item, seriesIndex) => (
          <g key={item.key}>
            <path
              d={model.paths[seriesIndex]}
              fill="none"
              stroke={item.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {model.points[seriesIndex].map((point) => (
              <circle
                key={`${item.key}-${point.bucket.start}`}
                cx={model.x(point.date)}
                cy={model.y(point.value)}
                r={hovered === point ? 5 : 3}
                fill="var(--surface)"
                stroke={item.color}
                strokeWidth="2"
                tabIndex={0}
                onPointerEnter={() => setHovered(point)}
                onPointerLeave={() => setHovered(null)}
                onFocus={() => setHovered(point)}
                onBlur={() => setHovered(null)}
              >
                <title>{`${item.label}: ${point.value} · ${bucketLabel(point.bucket)}`}</title>
              </circle>
            ))}
          </g>
        ))}

        {hovered ? (
          <g pointerEvents="none">
            <rect
              x={tooltipLeft}
              y={Math.max(MARGIN.top, tooltipY - 55)}
              width={tooltipWidth}
              height="42"
              rx="5"
              fill="var(--surface)"
              stroke="var(--line-strong)"
            />
            <text
              x={tooltipLeft + 8}
              y={Math.max(MARGIN.top, tooltipY - 55) + 15}
              fill="var(--ink)"
              fontSize="11"
              fontWeight="600"
            >
              {hovered.series.label}: {hovered.value}
            </text>
            <text
              x={tooltipLeft + 8}
              y={Math.max(MARGIN.top, tooltipY - 55) + 31}
              fill="var(--ink-3)"
              fontSize="10"
            >
              {bucketLabel(hovered.bucket)}
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}
