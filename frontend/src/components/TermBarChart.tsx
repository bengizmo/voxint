import { scaleLinear } from "d3-scale";

import type { TermDatum } from "./WordCloud";

interface TermBarChartProps {
  terms: TermDatum[];
  onTermClick: (term: string) => void;
  maxItems?: number;
}

export function TermBarChart({
  terms,
  onTermClick,
  maxItems = 20,
}: TermBarChartProps) {
  const items = terms.slice(0, maxItems);
  if (!items.length) return null;

  const maxCount = Math.max(...items.map((t) => t.count));
  const barScale = scaleLinear().domain([0, maxCount]).range([0, 100]);

  return (
    <div
      className="term-bar-chart"
      role="list"
      aria-label="Top terms by frequency"
    >
      {items.map((t) => (
        <button
          key={t.term}
          className="term-bar-row"
          onClick={() => onTermClick(t.term)}
          role="listitem"
          type="button"
        >
          <span className="term-bar-label">{t.term}</span>
          <span className="term-bar-track">
            <span
              className="term-bar-fill"
              style={{ width: `${barScale(t.count)}%` }}
            />
          </span>
          <span className="term-bar-count">{t.count}</span>
        </button>
      ))}
    </div>
  );
}
