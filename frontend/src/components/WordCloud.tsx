import { useEffect, useMemo, useRef, useState } from "react";
import cloud, { type Word } from "d3-cloud";
import { scaleLinear, scaleSqrt } from "d3-scale";

export interface TermDatum {
  term: string;
  count: number;
  doc_count: number;
  tfidf: number;
}

interface CloudWord extends Word {
  tfidf: number;
  count: number;
}

interface PositionedWord {
  text: string;
  size: number;
  x: number;
  y: number;
  rotate: number;
  tfidf: number;
  count: number;
}

interface WordCloudProps {
  terms: TermDatum[];
  onTermClick: (term: string) => void;
  width?: number;
  height?: number;
}

export function WordCloud({
  terms,
  onTermClick,
  width = 480,
  height = 280,
}: WordCloudProps) {
  const [words, setWords] = useState<PositionedWord[]>([]);
  const layoutRef = useRef<ReturnType<typeof cloud> | null>(null);
  const displayed = useMemo(() => terms.slice(0, 80), [terms]);
  const termsKey = useMemo(
    () => displayed.map((t) => `${t.term}:${t.count}:${t.tfidf}`).join(","),
    [displayed],
  );

  useEffect(() => {
    if (!terms.length) return;

    const counts = displayed.map((t) => t.count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);
    const fontSize: (n: number) => number =
      minCount === maxCount
        ? () => 20
        : scaleSqrt().domain([minCount, maxCount]).range([10, 36]).clamp(true);

    const input: CloudWord[] = displayed.map((t) => ({
      text: t.term,
      size: fontSize(t.count),
      tfidf: t.tfidf,
      count: t.count,
    }));

    if (layoutRef.current) {
      layoutRef.current.stop();
    }

    const layout = cloud()
      .size([width, height])
      .words(input)
      .padding(3)
      .rotate(() => 0)
      .font("IBM Plex Sans, system-ui, sans-serif")
      .fontSize((d) => d.size ?? 10)
      .on("end", (positioned) => {
        setWords(
          positioned.map((w) => ({
            text: w.text ?? "",
            size: w.size ?? 10,
            x: w.x ?? 0,
            y: w.y ?? 0,
            rotate: w.rotate ?? 0,
            tfidf: (w as CloudWord).tfidf,
            count: (w as CloudWord).count,
          })),
        );
      });

    layoutRef.current = layout;
    layout.start();

    return () => {
      layout.stop();
    };
  }, [termsKey, width, height]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!terms.length) return null;

  const tfidfValues = words.map((w) => w.tfidf);
  const maxTfidf = tfidfValues.length ? Math.max(...tfidfValues) : 1;
  const minTfidf = tfidfValues.length ? Math.min(...tfidfValues) : 0;
  const opacity: (n: number) => number =
    minTfidf === maxTfidf
      ? () => 1
      : scaleLinear().domain([minTfidf, maxTfidf]).range([0.4, 1]);

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="word-cloud-svg"
      role="group"
      aria-label="Word cloud of top corpus terms"
    >
      <g transform={`translate(${width / 2},${height / 2})`}>
        {words.map((w) => (
          <text
            key={w.text}
            textAnchor="middle"
            transform={`translate(${w.x},${w.y}) rotate(${w.rotate})`}
            style={{
              fontSize: `${w.size}px`,
              fontFamily: "IBM Plex Sans, system-ui, sans-serif",
              fill: "var(--accent)",
              opacity: opacity(w.tfidf),
              cursor: "pointer",
            }}
            onClick={() => onTermClick(w.text)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") onTermClick(w.text);
            }}
          >
            <title>
              {w.text}: {w.count} occurrences
            </title>
            {w.text}
          </text>
        ))}
      </g>
    </svg>
  );
}
