import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";

export interface TimelineInterval {
  start_seconds: number;
  end_seconds: number;
  label: string;
  speaker_name: string | null;
  speaker_id: string | null;
  resolution: string;
  overlap: boolean;
}

export interface TimelineLane {
  label: string;
  speaker_name: string | null;
  speaker_id: string | null;
  resolution: string;
  total_seconds: number;
  turn_count: number;
  intervals: TimelineInterval[];
}

export interface SpeakerTimeline {
  duration_seconds: number;
  speaker_count: number;
  lanes: TimelineLane[];
}

interface SpeakerTimelineIslandProps {
  runId: string;
  timeline: SpeakerTimeline;
}

interface HoveredInterval {
  interval: TimelineInterval;
  lane: TimelineLane;
}

const FALLBACK_WIDTH = 760;
const MIN_WIDTH = 360;
const LABEL_WIDTH = 132;
const RIGHT_MARGIN = 12;
const TOP_MARGIN = 12;
const LANE_HEIGHT = 24;
const LANE_GAP = 4;
const BLOCK_INSET = 3;
const AXIS_HEIGHT = 34;
const LARGE_INTERVAL_COUNT = 1_000;

function speakerPaletteIndex(label: string): number {
  let hash = 0;
  for (const character of label) {
    hash = (hash * 31 + character.charCodeAt(0)) | 0;
  }
  return Math.abs(hash) % 8;
}

function formatTime(seconds: number, long: boolean): string {
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3_600);
  const minutes = Math.floor((rounded % 3_600) / 60);
  const remainder = rounded % 60;
  if (long || hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function intervalName(lane: TimelineLane): string {
  if (lane.resolution === "human_exclude") return `${lane.label} (excluded)`;
  return lane.speaker_name ?? lane.label;
}

function intervalAriaLabel(
  lane: TimelineLane,
  interval: TimelineInterval,
  long: boolean,
): string {
  const duration = Math.max(0, interval.end_seconds - interval.start_seconds);
  return `${intervalName(lane)}, ${formatTime(interval.start_seconds, long)} to ${formatTime(interval.end_seconds, long)}, ${duration.toFixed(1)} seconds${interval.overlap ? ", overlapping speech" : ""}`;
}

export function SpeakerTimelineIsland({
  runId,
  timeline,
}: SpeakerTimelineIslandProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FALLBACK_WIDTH);
  const [hovered, setHovered] = useState<HoveredInterval | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const updateWidth = (nextWidth: number) => {
      if (nextWidth > 0) setWidth(Math.max(MIN_WIDTH, Math.round(nextWidth)));
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
    const duration = Math.max(0.001, timeline.duration_seconds);
    const plotStart = Math.min(LABEL_WIDTH, width * 0.38);
    const plotEnd = Math.max(plotStart + 1, width - RIGHT_MARGIN);
    const x = scaleLinear().domain([0, duration]).range([plotStart, plotEnd]);
    const intervalCount = timeline.lanes.reduce(
      (total, lane) => total + lane.intervals.length,
      0,
    );
    let filteredCount = 0;
    const lanes = timeline.lanes.map((lane, laneIndex) => {
      const visible = lane.intervals.filter((interval) => {
        if (
          intervalCount > LARGE_INTERVAL_COUNT &&
          x(interval.end_seconds) - x(interval.start_seconds) < 1
        ) {
          filteredCount++;
          return false;
        }
        return true;
      });
      return {
        lane,
        color: `var(--spk-${speakerPaletteIndex(lane.speaker_id ?? lane.label)})`,
        patternId: `speaker-overlap-${laneIndex}`,
        intervals: visible,
      };
    });
    const axisY =
      TOP_MARGIN + timeline.lanes.length * (LANE_HEIGHT + LANE_GAP) + 2;
    return {
      x,
      lanes,
      axisY,
      height: axisY + AXIS_HEIGHT,
      ticks: x.ticks(
        Math.max(2, Math.min(7, Math.floor((plotEnd - plotStart) / 90))),
      ),
      longTime: timeline.duration_seconds >= 3_600,
      plotStart,
      plotEnd,
      filteredCount,
    };
  }, [timeline, width]);

  if (!timeline.lanes.length) {
    return <p className="oc-muted">No diarization data available</p>;
  }

  const hoveredX = hovered
    ? model.x(
        (hovered.interval.start_seconds + hovered.interval.end_seconds) / 2,
      )
    : 0;
  const tooltipWidth = 218;
  const tooltipLeft = Math.min(
    Math.max(model.plotStart, hoveredX - tooltipWidth / 2),
    model.plotEnd - tooltipWidth,
  );

  return (
    <section aria-labelledby="speaker-timeline-heading">
      <h2 id="speaker-timeline-heading">Speaker timeline</h2>
      <div ref={containerRef} className="w-full overflow-x-auto">
        <svg
          className="block w-full"
          viewBox={`0 0 ${width} ${model.height}`}
          role="img"
          aria-label={`Speaker timeline with ${timeline.lanes.length} lanes over ${formatTime(timeline.duration_seconds, model.longTime)}`}
        >
          <title>Speaker activity over the recording</title>
          <defs>
            {model.lanes.map(({ lane, color, patternId }) => (
              <pattern
                id={patternId}
                key={`pattern-${lane.label}`}
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <rect width="6" height="6" fill={color} />
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="6"
                  stroke="var(--surface)"
                  strokeWidth="2"
                  opacity="0.65"
                />
              </pattern>
            ))}
          </defs>

          {model.lanes.map(
            ({ lane, intervals, color, patternId }, laneIndex) => {
              const laneY = TOP_MARGIN + laneIndex * (LANE_HEIGHT + LANE_GAP);
              const excluded = lane.resolution === "human_exclude";
              const unresolved = lane.resolution === "unresolved";
              return (
                <g key={lane.label} opacity={excluded ? 0.42 : 1}>
                  <text
                    x={model.plotStart - 8}
                    y={laneY + LANE_HEIGHT / 2}
                    dy="0.34em"
                    textAnchor="end"
                    fill={excluded ? "var(--ink-3)" : "var(--ink-2)"}
                    fontSize="11"
                  >
                    {intervalName(lane)}
                  </text>
                  <line
                    x1={model.plotStart}
                    x2={model.plotEnd}
                    y1={laneY + LANE_HEIGHT / 2}
                    y2={laneY + LANE_HEIGHT / 2}
                    stroke="var(--line)"
                  />
                  {intervals.map((interval, intervalIndex) => {
                    const startX = model.x(interval.start_seconds);
                    const blockWidth = Math.max(
                      0.75,
                      model.x(interval.end_seconds) - startX,
                    );
                    const label = intervalAriaLabel(
                      lane,
                      interval,
                      model.longTime,
                    );
                    return (
                      <a
                        key={`${interval.start_seconds}-${interval.end_seconds}-${intervalIndex}`}
                        href={`/runs/${runId}/transcript#t=${interval.start_seconds}`}
                        aria-label={`${label}. Open transcript at this time.`}
                        onPointerEnter={() => setHovered({ lane, interval })}
                        onPointerLeave={() => setHovered(null)}
                        onFocus={() => setHovered({ lane, interval })}
                        onBlur={() => setHovered(null)}
                      >
                        <rect
                          x={startX}
                          y={laneY + BLOCK_INSET}
                          width={blockWidth}
                          height={LANE_HEIGHT - BLOCK_INSET * 2}
                          rx="2"
                          fill={interval.overlap ? `url(#${patternId})` : color}
                          fillOpacity={unresolved ? 0.52 : 0.88}
                          stroke={unresolved ? color : "none"}
                          strokeWidth={unresolved ? 1.5 : 0}
                          strokeDasharray={unresolved ? "3 2" : undefined}
                        >
                          <title>{label}</title>
                        </rect>
                      </a>
                    );
                  })}
                </g>
              );
            },
          )}

          <line
            x1={model.plotStart}
            x2={model.plotEnd}
            y1={model.axisY}
            y2={model.axisY}
            stroke="var(--line-strong)"
          />
          {model.ticks.map((tick) => (
            <g key={tick} transform={`translate(${model.x(tick)},0)`}>
              <line
                y1={model.axisY}
                y2={model.axisY + 5}
                stroke="var(--line-strong)"
              />
              <text
                y={model.axisY + 18}
                textAnchor="middle"
                fill="var(--ink-3)"
                fontSize="10"
              >
                {formatTime(tick, model.longTime)}
              </text>
            </g>
          ))}

          {hovered ? (
            <g pointerEvents="none">
              <rect
                x={tooltipLeft}
                y={Math.max(2, model.axisY - 50)}
                width={tooltipWidth}
                height="42"
                rx="5"
                fill="var(--surface)"
                stroke="var(--line-strong)"
              />
              <text
                x={tooltipLeft + 8}
                y={Math.max(2, model.axisY - 50) + 15}
                fill="var(--ink)"
                fontSize="11"
                fontWeight="600"
              >
                {intervalName(hovered.lane)}
              </text>
              <text
                x={tooltipLeft + 8}
                y={Math.max(2, model.axisY - 50) + 31}
                fill="var(--ink-3)"
                fontSize="10"
              >
                {formatTime(hovered.interval.start_seconds, model.longTime)} –{" "}
                {formatTime(hovered.interval.end_seconds, model.longTime)} ·{" "}
                {(
                  hovered.interval.end_seconds - hovered.interval.start_seconds
                ).toFixed(1)}
                s
              </text>
            </g>
          ) : null}
        </svg>
      </div>

      {model.filteredCount > 0 && (
        <p
          className="text-xs"
          style={{ color: "var(--ink-3)", marginTop: "0.25rem" }}
        >
          {model.filteredCount} very short turns not shown at this scale.
        </p>
      )}

      <div
        className="mt-2 flex flex-wrap gap-x-4 gap-y-1"
        aria-label="Speaker timeline legend"
      >
        {model.lanes.map(({ lane, color }) => (
          <span
            key={lane.label}
            className="inline-flex items-center gap-1.5 text-[length:var(--t-xs)] text-[var(--ink-2)]"
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{
                backgroundColor: color,
                opacity: lane.resolution === "human_exclude" ? 0.42 : 0.88,
                border:
                  lane.resolution === "unresolved"
                    ? `1px dashed ${color}`
                    : undefined,
              }}
              aria-hidden="true"
            />
            {intervalName(lane)} ·{" "}
            {formatTime(lane.total_seconds, model.longTime)}
          </span>
        ))}
      </div>
    </section>
  );
}
