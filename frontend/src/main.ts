// One shared entry every page pulls in (base.html). Does nothing if a page has
// no [data-island] nodes. Dynamically imports only the island bundles present,
// keeping this file tiny and letting future issues add islands without touching
// base.html.

import { isPaletteChord } from "./lib/palette";

type MountFn = (el: HTMLElement, opts?: { open?: boolean }) => void;
type Loader = () => Promise<{ mount: MountFn }>;

const registry: Record<string, Loader> = {
  "transcript-player": () => import("./entries/transcript-player"),
  "corrections-editor": () => import("./entries/corrections-editor"),
  "media-editor": () => import("./entries/media-editor"),
  "explore": () => import("./entries/explore"),
  "quote-board": () => import("./entries/quote-board"),
  "temporal-trends": () => import("./entries/temporal-trends"),
  "speaker-timeline": () => import("./entries/speaker-timeline"),
  "command-palette": () => import("./entries/command-palette"),
};

// Arm a lazy island: defer import until the first click on the element or
// the first Ctrl/Cmd+K chord. This keeps the palette bundle out of pages
// that never open it.
function armLazyIsland(el: HTMLElement, load: Loader): void {
  let armed = true;

  const trigger = (open: boolean) => {
    if (!armed) return;
    armed = false;
    el.removeEventListener("click", onClick);
    window.removeEventListener("keydown", onKeyDown);
    load()
      .then((mod) => mod.mount(el, { open }))
      .catch((err: unknown) => {
        console.error(`lazy island failed to hydrate`, err);
      });
  };

  const onClick = () => trigger(true);
  const onKeyDown = (event: KeyboardEvent) => {
    if (isPaletteChord(event)) {
      event.preventDefault();
      trigger(true);
    }
  };

  el.addEventListener("click", onClick);
  window.addEventListener("keydown", onKeyDown);
}

for (const el of document.querySelectorAll<HTMLElement>("[data-island]")) {
  const name = el.dataset.island;
  const load = name ? registry[name] : undefined;
  if (!load) continue;

  if (el.hasAttribute("data-island-lazy")) {
    armLazyIsland(el, load);
  } else {
    load()
      .then((mod) => mod.mount(el))
      .catch((err: unknown) => {
        // Island failure degrades ONE region, never the page. Server-rendered
        // fallback markup inside the div stays visible.
        console.error(`island "${name}" failed to hydrate`, err);
      });
  }
}
