import { scaleLinear } from "d3-scale";

import type { TermDatum } from "./WordCloud";

interface TermBarChartProps {
  terms: TermDatum[];
  // Omit to render non-interactive rows (e.g. the tag rollup, which has no
  // drill-down target).
  onTermClick?: (term: string) => void;
  maxItems?: number;
  ariaLabel?: string;
}

export function TermBarChart({
  terms,
  onTermClick,
  maxItems = 20,
  ariaLabel = "Top terms by frequency",
}: TermBarChartProps) {
  const items = terms.slice(0, maxItems);
  if (!items.length) return null;

  const maxCount = Math.max(...items.map((t) => t.count));
  const barScale = scaleLinear().domain([0, maxCount]).range([0, 100]);

  return (
    <div className="term-bar-chart" aria-label={ariaLabel}>
      {items.map((t) => {
        const rowContent = (
          <>
            <span className="term-bar-label">{t.term}</span>
            <span className="term-bar-track">
              <span
                className="term-bar-fill"
                style={{ width: `${barScale(t.count)}%` }}
              />
            </span>
            <span className="term-bar-count">{t.count}</span>
          </>
        );
        return onTermClick ? (
          <button
            key={t.term}
            className="term-bar-row"
            onClick={() => onTermClick(t.term)}
            type="button"
          >
            {rowContent}
          </button>
        ) : (
          <div key={t.term} className="term-bar-row">
            {rowContent}
          </div>
        );
      })}
    </div>
  );
}
