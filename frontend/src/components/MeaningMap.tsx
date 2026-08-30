import { useCallback, useEffect, useRef, useState } from "react";

/** Semantic meaning map (#357): a canvas scatter of the corpus chunk
 * embeddings projected to 2D server-side (PCA). Points are colored by
 * recording; hover shows a tooltip, click selects, and the selected passage's
 * details render as ordinary HTML below the canvas so the transcript link is
 * never canvas-only. Data loads lazily from /explore/meaning-map when the
 * section is first opened — never inlined in island props. */

export interface MapPoint {
  x: number;
  y: number;
  run_id: string;
  media_title: string;
  speaker_label: string | null;
  start_seconds: number;
  end_seconds: number;
  preview: string;
  jump_url: string;
}

interface MapResponse {
  state: "ok" | "off" | "insufficient";
  points?: MapPoint[];
  total_n?: number;
  shown_n?: number;
  sampled?: boolean;
}

type FetchState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error" }
  | { phase: "ready"; data: MapResponse };

const HEIGHT = 420;
const PAD = 24;
const POINT_RADIUS = 3.5;
const HIT_RADIUS = 10;

function paletteIndex(runId: string): number {
  let hash = 0;
  for (const character of runId) hash = (hash * 31 + character.charCodeAt(0)) | 0;
  return Math.abs(hash) % 8;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Screen-space projection shared by the draw loop and hit-testing: one
 * transform, in CSS pixels, so a DPR-scaled backing store cannot skew the
 * hover math. */
function projector(points: MapPoint[], width: number, height: number) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const innerW = width - PAD * 2;
  const innerH = height - PAD * 2;
  return (p: MapPoint): [number, number] => [
    PAD + ((p.x - minX) / spanX) * innerW,
    // Flip y: PCA "up" should read as up on screen.
    PAD + (1 - (p.y - minY) / spanY) * innerH,
  ];
}

export function MeaningMap({ projectId }: { projectId: string | null }) {
  const [open, setOpen] = useState(false);
  const [fetchState, setFetchState] = useState<FetchState>({ phase: "idle" });
  const [hovered, setHovered] = useState<{ point: MapPoint; sx: number; sy: number } | null>(null);
  const [selected, setSelected] = useState<MapPoint | null>(null);
  const [width, setWidth] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(async () => {
    setFetchState({ phase: "loading" });
    try {
      const qs = projectId ? `?project=${encodeURIComponent(projectId)}` : "";
      const res = await fetch(`/explore/meaning-map${qs}`);
      if (!res.ok) {
        setFetchState({ phase: "error" });
        return;
      }
      setFetchState({ phase: "ready", data: (await res.json()) as MapResponse });
    } catch {
      setFetchState({ phase: "error" });
    }
  }, [projectId]);

  const toggle = useCallback(() => {
    setOpen((was) => {
      if (!was && fetchState.phase === "idle") void load();
      return !was;
    });
  }, [fetchState.phase, load]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [open]);

  const points =
    fetchState.phase === "ready" && fetchState.data.state === "ok"
      ? (fetchState.data.points ?? [])
      : [];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width === 0 || points.length === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(HEIGHT * dpr);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, HEIGHT);
    const project = projector(points, width, HEIGHT);
    const colors = Array.from({ length: 8 }, (_, i) => cssVar(`--spk-${i}`));
    context.globalAlpha = 0.6;
    for (const p of points) {
      const [sx, sy] = project(p);
      context.beginPath();
      context.arc(sx, sy, POINT_RADIUS, 0, Math.PI * 2);
      context.fillStyle = colors[paletteIndex(p.run_id)] || cssVar("--accent");
      context.fill();
    }
    if (selected) {
      context.globalAlpha = 1;
      const [sx, sy] = project(selected);
      context.beginPath();
      context.arc(sx, sy, POINT_RADIUS + 2.5, 0, Math.PI * 2);
      context.strokeStyle = cssVar("--ink");
      context.lineWidth = 1.5;
      context.stroke();
    }
  }, [points, width, selected]);

  const nearestPoint = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || points.length === 0) return null;
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const project = projector(points, width, HEIGHT);
      let best: MapPoint | null = null;
      let bestDist = HIT_RADIUS * HIT_RADIUS;
      let bestXY: [number, number] = [0, 0];
      for (const p of points) {
        const [sx, sy] = project(p);
        const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        if (d < bestDist) {
          bestDist = d;
          best = p;
          bestXY = [sx, sy];
        }
      }
      return best ? { point: best, sx: bestXY[0], sy: bestXY[1] } : null;
    },
    [points, width],
  );

  const handleMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      setHovered(nearestPoint(event));
    },
    [nearestPoint],
  );

  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const hit = nearestPoint(event);
      setSelected(hit ? hit.point : null);
    },
    [nearestPoint],
  );

  return (
    <section className="mb-4" aria-label="Meaning map">
      <button
        type="button"
        className="secondary"
        onClick={toggle}
        aria-expanded={open}
      >
        {open ? "Hide meaning map" : "Show meaning map"}
      </button>

      {open ? (
        <div className="mt-3">
          {fetchState.phase === "loading" ? (
            <p className="text-sm text-[var(--ink-3)]">Building the map…</p>
          ) : fetchState.phase === "error" ? (
            <p className="text-sm text-[var(--ink-3)]">
              The map could not load. Try again after a refresh.
            </p>
          ) : fetchState.phase === "ready" && fetchState.data.state === "off" ? (
            <p className="text-sm text-[var(--ink-3)]">
              The semantic index is disabled, so there is no map to draw. Enable
              it in Settings to use this view.
            </p>
          ) : fetchState.phase === "ready" &&
            fetchState.data.state === "insufficient" ? (
            <p className="text-sm text-[var(--ink-3)]">
              Not enough indexed passages yet for a useful map. It appears once a
              few transcripts have been processed.
            </p>
          ) : fetchState.phase === "ready" ? (
            <>
              <div ref={wrapRef} className="relative">
                <canvas
                  ref={canvasRef}
                  style={{ width: "100%", height: HEIGHT, display: "block" }}
                  className="rounded-[var(--r-md)] border border-[var(--line)] bg-[var(--surface)]"
                  role="img"
                  aria-label={`Meaning map: ${points.length} passages positioned by similarity`}
                  onMouseMove={handleMove}
                  onMouseLeave={() => setHovered(null)}
                  onClick={handleClick}
                />
                {hovered ? (
                  <div
                    className="pointer-events-none absolute z-10 max-w-[32ch] rounded-[var(--r-sm)] border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[length:var(--t-micro)] shadow-sm"
                    style={{
                      left: Math.min(hovered.sx + 10, Math.max(width - 260, 0)),
                      top: hovered.sy + 10,
                    }}
                  >
                    <span className="block font-semibold text-[var(--ink)]">
                      {hovered.point.media_title} ·{" "}
                      {formatTime(hovered.point.start_seconds)}
                    </span>
                    <span className="block text-[var(--ink-2)]">
                      {hovered.point.preview}
                    </span>
                  </div>
                ) : null}
              </div>
              <p className="mt-1 text-[length:var(--t-micro)] text-[var(--ink-3)]">
                Each dot is a passage, colored by recording. Nearby dots use
                similar language; the layout is approximate and recalculates as
                the corpus grows.
                {fetchState.data.sampled
                  ? ` Showing ${fetchState.data.shown_n} of ${fetchState.data.total_n} passages.`
                  : ""}
              </p>
              {selected ? (
                <div className="mt-2 rounded-[var(--r-md)] border border-[var(--line)] bg-[var(--surface)] px-3 py-2 text-sm">
                  <span className="block text-[length:var(--t-micro)] text-[var(--ink-3)]">
                    {selected.speaker_label ? `${selected.speaker_label} · ` : ""}
                    {selected.media_title} · {formatTime(selected.start_seconds)}
                  </span>
                  <span className="block text-[var(--ink-2)]">
                    {selected.preview}
                  </span>
                  <a
                    className="text-[var(--accent)] no-underline hover:underline"
                    href={selected.jump_url}
                  >
                    Open in transcript
                  </a>
                </div>
              ) : points.length > 0 ? (
                <p className="mt-2 text-[length:var(--t-micro)] text-[var(--ink-3)]">
                  Click a dot to see its passage and jump to the transcript.
                </p>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
