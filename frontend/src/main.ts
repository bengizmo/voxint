// One shared entry every page pulls in (base.html). Does nothing if a page has
// no [data-island] nodes. Dynamically imports only the island bundles present,
// keeping this file tiny and letting future issues add islands without touching
// base.html.
const registry: Record<string, () => Promise<{ mount: (el: HTMLElement) => void }>> = {
  "transcript-player": () => import("./entries/transcript-player"),
  "workbench-player": () => import("./entries/workbench-player"),
  "review-stepper": () => import("./entries/review-stepper"),
};

for (const el of document.querySelectorAll<HTMLElement>("[data-island]")) {
  const name = el.dataset.island;
  const load = name ? registry[name] : undefined;
  if (!load) continue;
  load()
    .then((mod) => mod.mount(el))
    .catch((err: unknown) => {
      // Island failure degrades ONE region, never the page. Server-rendered
      // fallback markup inside the div stays visible.
      console.error(`island "${name}" failed to hydrate`, err);
    });
}
