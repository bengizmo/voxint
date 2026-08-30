import { useEffect, useMemo, useRef, useState } from "react";
import cloud, { type Word } from "d3-cloud";
import { scaleLinear } from "d3-scale";

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
  const termsKey = useMemo(
    () => terms.map((t) => `${t.term}:${t.tfidf}`).join(","),
    [terms],
  );

  useEffect(() => {
    if (!terms.length) return;

    const maxTfidf = Math.max(...terms.map((t) => t.tfidf));
    const minTfidf = Math.min(...terms.map((t) => t.tfidf));
    const fontSize = scaleLinear()
      .domain([minTfidf, maxTfidf])
      .range([10, 36])
      .clamp(true);

    const input: CloudWord[] = terms.slice(0, 80).map((t) => ({
      text: t.term,
      size: fontSize(t.tfidf),
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

  const maxTfidf = Math.max(...words.map((w) => w.tfidf), 1);
  const opacity = scaleLinear().domain([0, maxTfidf]).range([0.4, 1]);

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
